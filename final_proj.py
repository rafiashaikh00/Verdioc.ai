
import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever
)
from langchain_classic.retrievers.document_compressors import (
    LLMChainExtractor
)

from langchain_ollama import ChatOllama

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel
)

from langchain_core.output_parsers import StrOutputParser


# tempfile is a way where we upload a file but that file not reached pypdf
# so store that file in temp path from their we sent this pdf to pypdf loader
# PyPDFLoader ko PDF haath mein dena nahi aata; usko address/path chahiye.
# Isliye temporary file bana kar uska path dete hain.

st.title("📖 Veridoc")

st.write("Chat with your PDF using local RAG.")


# jab humare pas chat history empty ho to ek empty list banao
# ok hum chat history ko session m store krty hn

if 'chat_history' not in st.session_state:

    st.session_state.chat_history = []


# agr main chain session m nhi hn to none create kro

if "main_chain" not in st.session_state:
  #we created this cause it work placeholder start mein which has none value after chain start firstq ues se none replace to rag entire chain main chain has everthing so after every ques not rag chain build again same one chain work for ur next swal s
    st.session_state.main_chain = None


# file upload box bane ga
# Ek correction: Streamlit script har interaction par rerun hoti hai,
# sirf upload file na hone par nahi.
# session_state ka purpose ye hai ke rerun ke bawajood important data preserve rahe

with st.sidebar:

    st.header("📑 Upload Pdf")

    upload_file = st.file_uploader(
        "Upload Your File",
        type=['pdf']
    )

    process_pdf = st.button("Click to Upload pdf")


    # agar new PDF process karni ho
    # to purani chat history aur purani main chain clear kar do

    if process_pdf:

        st.session_state.chat_history = []

        st.session_state.main_chain = None


    st.markdown("---")

    st.subheader("How it works")

    st.write(
        """
        1. Upload your PDF.
        2. Click "Process PDF".
        3. Ask Veridoc AI anything about your PDF.
        4. Get answers with references and page numbers.
        5. Press Exit when you're done.
        """
    )

    st.markdown("---")

    exit_button = st.button("Exit")

    if exit_button:

        st.session_state.chat_history = []

        st.session_state.main_chain = None

        st.stop()


# Ab RAG tabhi build hoga jab PDF upload ho
# aur Process PDF button click ho

if (
    process_pdf
    and upload_file is not None
):

    st.write("PDF uploaded successfully!")


    # tempfile ek lib hn python ki jo temp file ya folder banay ja sakty hn
    # with method NamedTemporaryFile se
    # as temp_file hum refer kare ge

    # mean as temp_file is refer name of temporary file which created

    with tempfile.NamedTemporaryFile(
        delete=False,
        # es ka matlab hn after with block ke baad bhi file exist kare
        # tak ke we can send this temp file path to pypdf loader

        # Temporary file ko automatically abhi delete mat karo;
        # mujhe uska path baad mein use karna hai.

        suffix='.pdf',
    ) as temp_file:

        # fir getbuffer ke bad us data ko temp file m save karo

        temp_file.write(
            upload_file.getbuffer()
            # uploaded PDF ka actual file data/bytes nikaalo.
        )


    path_pdf = temp_file.name
    # yha us ka address huga jo variable m store huga


    # ab es path ko loader ko dege hum

    loader = PyPDFLoader(path_pdf)

    # Streamlit se jo uploaded file mili, woh direct file-path nahi hai.
    # Hum uska data temporary .pdf file mein save karte hain,
    # phir us temporary file ka path PyPDFLoader ko dete hain.

    document = loader.load()
    # this actually load pdf


    # temporary file delete kar do
    os.remove(path_pdf)


    # stage 2 Text splitter

    split_text = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )


    chunk = split_text.split_documents(document)
    # document load go to txt splitter chunks bane


    # stage 3 embeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # stage 4 chroma

    vector_store = Chroma.from_documents(
        documents=chunk,
        embedding=embeddings,
        collection_name="proj1_new"
    )


    # stage 5 retrival

    retrival = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "fetch_k": 10,
            "k": 3,
            "lambda_mult": 1
        }
    )


    # stage 6 Model

    llm = ChatOllama(
        model="llama3.2:1b"
    )


    # STEG 7 cONETXT ANS JIS SE ANS PAGE COTENT AUR SHORT HUGA

    compreesion_llm = LLMChainExtractor.from_llm(llm)

    context_need_ret_llm = ContextualCompressionRetriever(
        base_retriever=retrival,
        base_compressor=compreesion_llm
    )
    # func jo swal leke doc lekar us ka context bane


    def prepare_rag_input(x):

        docs = context_need_ret_llm.invoke(
            x["question"]
        )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        reference = []
        page = []

        if docs:

            for doc in docs:

                reference.append(
                    doc.page_content
                )

                p = doc.metadata.get(
                    "page_label"
                )

                if p is None:

                    p = doc.metadata.get(
                        "page"
                    )

                if p is not None:

                    page.append(
                        str(p)
                    )

        return {
            "context": context,
            "question": x["question"],
            "chat_history": x["chat_history"],
            "reference": reference,
            "page": page
        }


    chat_template = ChatPromptTemplate([
        (
            "system",
            """
            You are a helpful PDF question-answering assistant.

            Use the provided PDF context to answer the user's question.

            Rules:
            - Understand the PDF context first.
            - Give the answer in your own words.
            - Do not copy the PDF text as the answer unless necessary.
            - The answer must be based only on the provided PDF context.
            - Do not add outside knowledge.
            - Do not make up information.
            - Use chat history to understand follow-up questions.
            - If the answer is not available in the provided PDF context, say:
            "The information is not available in the provided PDF."

            Give only the answer to the user's question.
            Do not mention the context, retrieval, page number, or reference.
            """
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        (
            "human",
            """
            PDF Context:
            {context}

            User Question:
            {question}
            """
        )
    ])


    chain_parallel = RunnableParallel({

        "answer":
            chat_template
            | llm
            | StrOutputParser(),

        "reference":
            RunnableLambda(
                lambda x: x["reference"]
            ),

        "page":
            RunnableLambda(
                lambda x: x["page"]
            )
    })


    # tak ke main chain save hu session state me
    # ave hu bar bar rerun na hu per user query message

    st.session_state.main_chain = (
        RunnableLambda(prepare_rag_input)
        |
        chain_parallel
    )


    st.success(
        "✅ PDF processed successfully! You can start chatting."
    )


# --------------------------------------------------
# Chat
# --------------------------------------------------

# chat input RAG build hone ke bahar hai
# kyun ke har question ke liye PDF ko dobara process nahi karna

if st.session_state.main_chain is not None:

    query = st.chat_input(
        "Ask Veridoc AI To Help You..."
    )
    # ask query chat mean puri chat hguei


    if query:
        # agr query yes hn


        # So with khud chat feature nahi hai.
        # st.chat_message() ek container deta hai,
        # aur with us container ke andar next Streamlit commands ko place karta hai.

        # Tumhari language mein:
        with st.chat_message("user"):
            # user container

            st.write(query)
            # wo query display act like print


        # chat jo le rhy hn us ko user ke container m show karna
        # fir neeche write se user query display


        result = st.session_state.main_chain.invoke({

            "question": query,

            "chat_history":
                st.session_state.chat_history
        })


        with st.chat_message("assistant"):

            st.write(
                result['answer']
            )


            if result['reference']:

                with st.expander(
                    "📌 Reference"
                ):

                    for ref in result['reference']:
                        # sare reference ek ek ref m ja ke print hu
                        # es liye we use loop
                        # jo original code m nhi tha

                        st.write(ref)


            if result["page"]:

                with st.expander(
                    "📄 Page"
                ):

                    st.write(
                        ", ".join(
                            result["page"]
                        )
                    )


        # chat history update

        st.session_state.chat_history.append(
            HumanMessage(
                content=query
            )
        )


        st.session_state.chat_history.append(
            AIMessage(
                content=result["answer"]
            )
        )


else:

    st.info(
        "👈 Upload your PDF from the sidebar "
        "and click 'Click to Upload pdf' to start."
    )

