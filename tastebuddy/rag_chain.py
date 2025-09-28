from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from tastebuddy.config import Config

class RAGChainBuilder:
    def __init__(self,vector_store):
        self.vector_store=vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL , temperature=0.5)
        self.history_store={}

    def _get_history(self,session_id:str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]
    
    def build_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k":3})

        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and user question, rewrite it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are TasteBuddy, an AI-powered culinary shopping assistant. 
        You answer using product reviews and titles, but ALWAYS format your response in **clean Markdown** for readability.  

        Formatting rules:
        - Use a **numbered list** for dishes or products.
        - Make dish/product names **bold**.
        - Add 1–2 relevant food emojis after each name (use your judgment).
        - Write a short engaging description on the next line.
        - Show ratings on their own line using stars (⭐) followed by the numeric score (e.g., ⭐⭐⭐⭐ 4.5/5).
        - On the next line, show reviews with the 📝 emoji and review count (e.g., 📝 1,200 reviews).
        - Before start listing next dish/product, add a blank line for better readability.
        - End with a friendly call-to-action encouraging user choice.

        IMPORTANT: Follow this exact example style:

        Example:

        1. **Lentil Mushroom Curry** 🍲👌  
        A flavorful and nutritious curry made with red or green lentils, mushrooms, and a blend of Indian spices.  

        ⭐⭐⭐⭐⭐ 4.9/5  
        📝 2,011 reviews  

        2. **Stuffed Portobello Mushrooms** 🍄🧀  
        Mushrooms filled with cheese, herbs, and breadcrumbs, baked to perfection.  

        ⭐⭐⭐⭐ 4.5/5  
        📝 800 reviews  

        👉 Which one would you like to try?

        CONTEXT:
        {context}
        """),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])


        history_aware_retriever = create_history_aware_retriever(
            self.model , retriever , context_prompt
        )

        question_answer_chain = create_stuff_documents_chain(
            self.model , qa_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,question_answer_chain
        )

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )


