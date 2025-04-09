from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import uuid
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()

TOKEN: Final = os.getenv("TOKEN")
BOT_USERNAME: Final = "@Simbolo_Bot"



async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Please Type Something")
    
async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("This is a custom command!")
    
async def final_answer_output(user_input: str) -> str:
    load_vector_db = FAISS.load_local(r"faiss_index", embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),allow_dangerous_deserialization=True)
    retriever  = load_vector_db.as_retriever()


    model = ChatGroq(temperature = 0.1, model_name = "llama-3.3-70b-versatile")

    retriever_prompt = (
        "Give a chat history and the latest user question which might reference context in the chat history,"
        "formulate a standalone question which can be understood without the chat history."
        "Do NOT answer the quesetion, just reformulate it if needed and otherwise return it as is."
        "Remember what the user ask"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", retriever_prompt),
            MessagesPlaceholder(variable_name = "chat_history"),
            ("human","{input}")
        ]
    )


    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

    system_message = """
    You are a professional and friendly AI assistant designed to provide excellent customer service. Your goal is to assist customers efficiently while maintaining a warm and empathetic tone.
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "We only answer about Simbolo." dont provide wrong answer. Always remember your answer must be related to Simbolo\n\n
    Context:\n {context}?\n
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_keys="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    session_id = str(uuid.uuid4())

    
    if user_input.lower() == "restart":
        store.clear()
    
    else:
        answer = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )["answer"]
    
    return answer

#async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
#    text = update.message.text
#    message = await generate_answer(text)  
#    await update.message.reply_text(message)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    print(f"Received message: {text}", flush=True)  # Debugging output
    await update.message.reply_text(await final_answer_output(text))

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update} caused error {context.error}")

if __name__ == "__main__":
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("custom", custom_command))
    
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    
    app.add_error_handler(error)
    print("Telegram bot has worked")
    app.run_polling(poll_interval=3)
