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


# -----------------------------
# 1. Load PDF
# -----------------------------

loader = PyPDFLoader("rag/coffee (1).pdf")

documents = loader.load()


# -----------------------------
# 2. Split PDF
# -----------------------------

split_text = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunk = split_text.split_documents(documents)


# -----------------------------
# 3. Embeddings
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# 4. Chroma
# -----------------------------

vector_store = Chroma.from_documents(
    documents=chunk,
    embedding=embeddings,
    collection_name="proj1_new"
)


# -----------------------------
# 5. Retriever
# -----------------------------

retrival = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "fetch_k": 10,
        "k": 3,
        "lambda_mult": 1
    }
)


# -----------------------------
# 6. LLM
# -----------------------------

llm = ChatOllama(
    model="llama3.2:1b"
)


# -----------------------------
# 7. Contextual Compression
# -----------------------------

compreesion_llm = LLMChainExtractor.from_llm(llm)

context_need_ret_llm = ContextualCompressionRetriever(
    base_retriever=retrival,
    base_compressor=compreesion_llm
)


# -----------------------------
# 8. Prompt
# -----------------------------

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
- If the answer is not available in the PDF context, say:
  "The information is not available in the provided PDF."

Give only the answer to the user's question.
Do not mention the context, retrieval, page number, or reference.
"""
    ),

    MessagesPlaceholder(variable_name="chat_history"),

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


# -----------------------------
# 9. Chat History
# -----------------------------

chat_history = []


# -----------------------------
# 10. Prepare RAG Input
# -----------------------------

def prepare_rag_input(x):
#x is act like a dict which takes two inputs one is ques other is history to us dict se only question nekalo
#ab es mein 3 doc hn doc mein jo page cotent per query k hesab se hn doc obj k tor pe
    docs = context_need_ret_llm.invoke(
        x["question"]
    )#output->  list of if3 doc object compressed
#ab sae 3 doc obj ka page cotent nekal k string m convert huga
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )#output -> string of pagecotent of 3 doc
#we are making an empty list which assign us value later 
    reference = []
    page = [] # we use varaible p cause list not compare ith none not none thats why store in p then add in append listwhat

    if docs:#  if list of compressed doc is not empty so
        #doc  ka page cotent jae ga list mein page wali lust mein
        for doc in docs:

          reference.append(doc.page_content)
        # doc  ke metadaat ka paglabel jae ga

          p=doc.metadata.get("page_label") # varaibel em rkho
    #backup line
          if p is None: 
            p=doc.metadata.get("page") # varaibel m pge num rkho
         #es se list m jae ga
          if p is not None:
            page.append(str(p)) # varibel ko list m dalo
  #es func me se ye chex jo bani wo retrun kare ga and x kyu k do inout le rha wo be retun
    return {
        "context": context,
        "question": x["question"],#same as it is wala invoke last m jo huwa history be same no change
        "chat_history": x["chat_history"],
        "reference": reference,
        "page": page
    }
#Input x ki useful cheezein + function mein banayi hui new information ko ek dictionary mein combine karke next chain ko dena.

# -----------------------------
# 11. Main Chain
# -----------------------------
#main chian ko history+ques mily  ga wo jae ga prep rag me us me se 5 cheeze jo retrun hui wo mily gei output mein
main_chain = (
    RunnableLambda(prepare_rag_input)#5 output pass huwe 

    |
   #ans ref page is key  runnable ko humesha dict form m input milta hn and us dict ka jo part nekalna hn jo lambda m use karo us ki eky ka nam likho
    RunnableParallel({
        "answer":# ab jo 5 input aya hn ye unh me se context question, history lega fill kar ga chat tmplate
            chat_template
            | llm
            | StrOutputParser(),#output -> llm ka ans

        "reference": # 5 input se ref nekalo lambda ek func bane ga wo input lega as x x me 5 input m se ref nekal k runnabel ko direct do
            RunnableLambda(lambda x: x["reference"]),# ref num

        "page":# 5 input se page do
            RunnableLambda(lambda x: x["page"]) # page num
    })
)#output -> llm ans,ref,page num, ye sabh neche as a dict result m jae ga


# -----------------------------
# 12. Chat Loop
# -----------------------------

while True:

    query = input("You: ")

    if query.lower() == "exit":
        break

    result = main_chain.invoke({
        "question": query,
        "chat_history": chat_history
    })# result have 

    print("\nAI:")
    print(result["answer"])

    print("\nReference:")
    print(result["reference"])

    print("\nPage:")
    print(result["page"])

    print()

    chat_history.append(
        HumanMessage(content=query)
    )

    chat_history.append(
        AIMessage(content=result["answer"])
)

