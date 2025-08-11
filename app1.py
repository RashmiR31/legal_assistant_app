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
from difflib import unified_diff
from html_templates import css, bot_template, user_template

# ---------------- PDF Processing ---------------- #
def extract_pdf_text_with_metadata(pdf_file):
    """Extracts all text from a single PDF with page metadata."""
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
    """Splits text into chunks and keeps metadata."""
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
    """Creates FAISS vectorstore with metadata."""
    embeddings = OpenAIEmbeddings()
    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

# ---------------- Conversation ---------------- #
def build_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0)
    custom_prompt = """
    You are a legal assistant AI. Answer based strictly on the provided context.
    - Always cite the exact clause, section, or page from the source document.
    - If the answer is not in the context, say: "The document does not provide this information."
    - Be concise but legally precise.
    - Ensure you provide all the details and a little more than what is asked in order to be helpful.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=custom_prompt, input_variables=["context", "question"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        output_key="answer"
    )

# ---------------- Document Comparison ---------------- #
def compare_all_documents(doc_texts):
    """Compares all uploaded documents pairwise and returns differences."""
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
        if i % 2 == 0:  # user
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:  # bot
            answer_text = message.content
            if sources and i == len(st.session_state.chat_history) - 1:
                source_info = "\n\n**Sources:**\n" + "\n".join(
                    [f"- {doc.metadata['source']} (page {doc.metadata['page']})" for doc in sources]
                )
                answer_text += source_info
            st.write(bot_template.replace("{{MSG}}", answer_text), unsafe_allow_html=True)
# ---------------- UI ---------------- #
def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({"question": user_question})
    st.session_state.chat_history = response['chat_history']
    sources = response.get('source_documents', [])

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # user
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:  # bot
            answer_text = message.content
            if sources and i == len(st.session_state.chat_history) - 1:
                source_info = "\n\n**Sources:**\n" + "\n".join(
                    [f"- {doc.metadata['source']} (page {doc.metadata['page']})" for doc in sources]
                )
                answer_text += source_info
            st.write(bot_template.replace("{{MSG}}", answer_text), unsafe_allow_html=True)

    # Auto-scroll to bottom of chat
    scroll_script = """
    <script>
        var chatContainer = window.parent.document.querySelector('.stChatMessageContainer');
        if (chatContainer) { chatContainer.scrollTop = chatContainer.scrollHeight; }
        window.scrollTo(0, document.body.scrollHeight);
    </script>
    """
    st.markdown(scroll_script, unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Legal Document Assistant", page_icon="⚖️")
    st.write(css, unsafe_allow_html=True)

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

if __name__ == "__main__":
    main()
