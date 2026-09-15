from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# 1.load the document

loader=PyPDFLoader('rag/coffee (1).pdf')
# doc obj create huge per page
documents=loader.load()
#2.split in to parts 
split_text=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
#at a time take one obj and split in to chunks
#chunk is a list of smaller doc ojects because it take one page ka doc object then divide in to chunks then chunk 0 is doc obj actual and chunk is list of all doc objects joa fter splitting form huwe
chunk=split_text.split_documents(documents)
#chunk variable contains a list of Document objects, where each Document represents one chunk.
"""print(len(chunk[0].page_content)) len of first chunk"""
#3.embeddings
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#for ... in ... inside [ ] → list comprehension, yani loop karke list banana.
"""b=[chunks.page_content for chunks in chunk]
a=embeddings.embed_documents(b)
print(len(a))# 6 doc obj in a list se 35 chunk bane per chunk 1 vector mean 35 total chunk and per vector ki 384 diemnsion so len of one chunk is also 384
print(len(a[0]))#384"""
#4 chroma add
vector_store=Chroma.from_documents(
    documents=chunk,
    embedding=embeddings,
    collection_name='proj1'
)
#step 5 convert in to retrival
#convert in to retrival
#Vector store ko retriever interface/capability de rahe hain, jisse woh query ke against relevant documents retrieve kar sake.
retrival=vector_store.as_retriever(
    search_type='mmr',
   #mmr work is query ke accordimg chunk hu and also repeated chunk na mily jo meaing zeyda same hu phly query k acc fetch k 10 chunk nekly ga acc to query then mmr and revelance and diversity dekh k top 3 dega
    search_kwargs={'fetch_k':10,'k':3,'lambda_mult':1}
)
query="what is coffee"
result=retrival.invoke(query)


#result contains a list of 3 retrieved Document objects.
#result is alist of 3 top docuemnts same link chunk is list w=if we wnat to see page content either we use loop to do see all or use indexing
"""print(result[0].page_content)#ye ese huga meri query ki embeddings chunk k jis embedding s match uga top 3 muje mily gei
print(len(result))
i=0
for doc in result:
    print("-----Cunk",i+1)
    print("-------",doc.page_content)
    print("------",doc.metadata)
    i=i+1"""
#step 6 retrival add
"""Compression LLM

Ab humein LLMChainExtractor ko ek LLM dena hai jo retrieved documents ko query ke according compress karega."""
llm=ChatOllama(model='llama3.2:1b')
#Hum ChatOllama wale LLM ko LLMChainExtractor ke through ye capability/task de rahe hain ke query ke according retrieved chunks mein se sirf relevant information extract kare aur unnecessary information remove kare."
compreesion_llm=LLMChainExtractor.from_llm(llm)
#es class ko retrival+llm chhye  ye varaibel obj hn es class ki retriver varaibel jese 
context_need_ret_llm=ContextualCompressionRetriever(
    #pehle relevant documents nikalta hai.
    base_retriever=retrival,
    #un documents ko query ke according compress karta hai.
    base_compressor=compreesion_llm
)
#es ka output hn ke compress doc list huga es k andr jo  doc obj huga us k pas page content m only query k kareb ans huga and meta data
compress_result=context_need_ret_llm.invoke(query)
#query go to conext need it go retriver it perform mmr result 3 doc go to llm chain it cut pagecontent to most revalnt ans of your query 
"""i=0
for doc in compress_result:
    print("-----doc",i+1)
    print("------- PAGE_CONTENT",doc.page_content)
    print("-----doc",i+1)
    print("------PAGE_METADATA",doc.metadata)
    i=i+1"""
# query -> retrival -> 3 doc -> all page content ko toor k query k heab se kare ga ab tak yhi hu rha

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful PDF-based question answering assistant.

Use the provided PDF context as the source of information for your answer.

Instructions:
- Identify the relevant facts and points from the provided context.
- Use those facts and points to construct your answer.
- Answer naturally in your own words instead of copying the context word-for-word.
- You may reorganize, combine, and explain the information to make the answer clearer.
- When helpful, you may give a simple example or analogy to explain the information from the context.
- Any factual information in your answer must be supported by the provided context.
- Do not use your general knowledge to add factual information that is not present in the context.
- Do not invent or assume information.
- If the requested information cannot be found in the provided context, clearly say:
  "The information is not available in the provided PDF."
- Do not pretend that unsupported information came from the PDF.
- The source page will be provided separately, so do not invent a page number.

Context:
{context}

Question:
{question}

Answer:
"""
)



# es se ye huga ke sare page content mil k ek string m convert huga
def fun_page_content_neklwao(x):
    result='\n\n'.join(content.page_content for content in x)
    return result
#context_result=fun_page_content_neklwao(compress_result) this si manual form later replace with chain smae content pripvide run parallel
run_parallel=RunnableParallel({
    'context':context_need_ret_llm|RunnableLambda(fun_page_content_neklwao),
    'question':RunnablePassthrough()
})
parser=StrOutputParser()
main_chain=run_parallel|prompt|llm|parser
res_all=main_chain.invoke(query)
print(res_all)
