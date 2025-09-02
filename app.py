import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain

from difflib import unified_diff
from html_templates import css, bot_template, user_template

# ---------------- LOGIN CONFIG ---------------- #
USER_CREDENTIALS = {
    "rashmi": "test123",
    "admin": "admin123"
}

def login_page():
    st.title("🔐 Login to Legal Assistant")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

def logout_button():
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()


# ---------------- PDF Processing ---------------- #
def extract_pdf_text_with_metadata(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    pages_data = []
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            pages_data.append({
                "text": text,
                "metadata": {
                    "source": pdf_file.name,
                    "page": i + 1
                }
            })
    return pages_data

def chunk_text_with_metadata(pages_data):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks_with_meta = []
    for page in pages_data:
        chunks = splitter.split_text(page["text"])
        for chunk in chunks:
            chunks_with_meta.append({
                "text": chunk,
                "metadata": page["metadata"]
            })
    return chunks_with_meta

def build_vectorstore(all_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

# ---------------- Conversation ---------------- #
# def build_conversation_chain(vectorstore):
#     llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")
#     custom_prompt = """
#     You are a legal assistant AI. Help a lawyer review the documents and give answers to their questions. Answer based strictly on the provided context.
#     - Always cite the exact clause, section, or page from the source document.
#     - Be concise but legally precise.
#     - Ensure you provide all the details and a little more than what is asked in order to be helpful.
#     - If the answer is not in the context, say: "The document does not provide this information."

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """
#     prompt = PromptTemplate(template=custom_prompt, input_variables=["context", "question"])
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

#     return ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
#         memory=memory,
#         combine_docs_chain_kwargs={"prompt": prompt},
#         return_source_documents=True,
#         output_key="answer"
#     )
def build_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    # Custom Legal Assistant Prompt

    custom_prompt = PromptTemplate(
    template="""
    You are a legal assistant AI. 
    Answer the question strictly using the information in the provided context. Help a lawyer review the documents and give answers to their questions. 

    Rules:
    - Only use the provided context to answer.
    - Always cite the page number and clause/section when possible.
    - Do not make assumptions or give generic legal advice.
    - Ensure you provide all the details and a little more than what is asked in order to be helpful.

    Context from the document:
    {context}

    Question:
    {question}

    Answer:
    """,
    input_variables=["context", "question"]
    )


    # Chain to insert context properly
    qa_chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        prompt=custom_prompt
    )

    # Question rephraser prompt
    question_gen_prompt = PromptTemplate(
        template="""Given the chat history and the follow up question,
rephrase the follow up question to be a standalone question.

Chat history:
{chat_history}
Follow up question: {question}
Standalone question:""",
        input_variables=["chat_history", "question"]
    )
    question_generator = LLMChain(llm=llm, prompt=question_gen_prompt)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    return ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
        combine_docs_chain=qa_chain,
        question_generator=question_generator,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

# ---------------- Document Comparison ---------------- #
def compare_all_documents(doc_texts):
    results = []
    filenames = list(doc_texts.keys())
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            f1, f2 = filenames[i], filenames[j]
            diff = unified_diff(
                doc_texts[f1].splitlines(),
                doc_texts[f2].splitlines(),
                fromfile=f1,
                tofile=f2,
                lineterm=''
            )
            diff_text = "\n".join(diff)
            results.append((f1, f2, diff_text if diff_text.strip() else "No differences found."))
    return results

# ---------------- UI ---------------- #
def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({"question": user_question})
    st.session_state.chat_history = response['chat_history']
    sources = response.get('source_documents', [])

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            answer_text = message.content
            if sources and i == len(st.session_state.chat_history) - 1:
                source_info = "\n\n**Sources:**\n" + "\n".join(
                    [f"- {doc.metadata['source']} (page {doc.metadata['page']})" for doc in sources]
                )
                answer_text += source_info
            st.write(bot_template.replace("{{MSG}}", answer_text), unsafe_allow_html=True)

    # Auto-scroll
    scroll_script = """
    <script>
        var chatContainer = window.parent.document.querySelector('.stChatMessageContainer');
        if (chatContainer) { chatContainer.scrollTop = chatContainer.scrollHeight; }
        window.scrollTo(0, document.body.scrollHeight);
    </script>
    """
    st.markdown(scroll_script, unsafe_allow_html=True)

# ---------------- MAIN ---------------- #
def main():
    load_dotenv()
    st.set_page_config(page_title="Legal Document Assistant", page_icon="⚖️", layout="wide")
    st.write(css, unsafe_allow_html=True)

    # LOGIN CHECK
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        login_page()
        return

    logout_button() 

    # Sidebar navigation
    page = st.sidebar.radio("Navigation", ["📄 Document Assistant", "💬 General Chat"])

    if page == "📄 Document Assistant":
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = None
        if "doc_texts" not in st.session_state:
            st.session_state.doc_texts = {}
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.header("Legal Document Q&A Assistant ⚖️")

        user_question = st.text_input("Ask a question about your uploaded legal documents:")
        if user_question and st.session_state.conversation:
            handle_userinput(user_question)

        with st.sidebar:
            st.subheader("Your Legal Documents")
            pdf_docs = st.file_uploader("Upload your legal PDFs", accept_multiple_files=True)

            if st.button("Process Documents"):
                with st.spinner("Processing..."):
                    all_chunks = []
                    doc_texts = {}
                    for pdf in pdf_docs:
                        pages_data = extract_pdf_text_with_metadata(pdf)
                        doc_texts[pdf.name] = "\n".join([p["text"] for p in pages_data])
                        chunks = chunk_text_with_metadata(pages_data)
                        all_chunks.extend(chunks)

                    st.session_state.doc_texts = doc_texts
                    st.session_state.vectorstore = build_vectorstore(all_chunks)
                    st.session_state.conversation = build_conversation_chain(st.session_state.vectorstore)
                    st.session_state.chat_history = []
                    st.success("Your legal assistant is ready.")

            if st.button("Compare Documents"):
                if len(st.session_state.doc_texts) >= 2:
                    comparisons = compare_all_documents(st.session_state.doc_texts)
                    for f1, f2, diff in comparisons:
                        st.subheader(f"Differences: {f1} vs {f2}")
                        st.code(diff, language="diff")
                else:
                    st.warning("Please upload at least two PDFs for comparison.")

            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                if st.session_state.vectorstore:
                    st.session_state.conversation = build_conversation_chain(st.session_state.vectorstore)
                st.success("Chat cleared. You can start a new conversation.")

    elif page == "💬 General Chat":
        st.header("⚖️ Legal Chat Assistant")

        if "general_chat" not in st.session_state:
            st.session_state.general_chat = []

        user_input = st.chat_input("Type your message...")

        if user_input:
            st.session_state.general_chat.append({"role": "user", "content": user_input})

            # Define system prompt
            system_prompt = """You are a professionally trained Indian legal drafting assistant. Your job is to independently draft high-quality legal documents that are legally compliant under Indian laws, including the Transfer of Property Act, 1882, Indian Contract Act, 1872, Registration Act, 1908, and relevant state stamp duty laws.

When asked to draft a document (e.g., Sale Deed, Legal Notice, Affidavit, Agreement, Petition), generate a detailed and fully formatted version, including all necessary clauses, declarations, and legal safeguards.

Always ensure:

Use of proper headings, alignment, and legal formatting.

Inclusion of relevant statutory references.

Inclusion of property schedule (with directions and measurements, if applicable).

Use of formal legal English, suitable for Indian registrars and courts.

Avoid placeholders like "[Name]" — generate realistic sample names or clearly mark optional fields.

Do not generate short templates or outlines. Provide full drafts that reduce the lawyer's drafting time.            """

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{question}")
            ])

            llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

            # Format prompt with user question
            formatted_prompt = prompt_template.format(question=user_input)

            # Get AI's text content only
            ai_response = llm.invoke(formatted_prompt).content.strip()

            st.session_state.general_chat.append({"role": "assistant", "content": ai_response})

        if st.button("🗑️ Clear General Chat"):
            st.session_state.general_chat = []
            st.rerun()

        # Display messages
        for msg in st.session_state.general_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])


if __name__ == "__main__":
    main()
