import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

load_dotenv()

print("Initializing Components....")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4.1-mini")

vector_store = Chroma(
    collection_name="medium_blogs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

retriever = vector_store.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    
    {context}

    Question: {question}

    Provide a detailed answer: ""

    """
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query:str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them and generates a response.

    Limitations:
    - Manual step by step execution.
    - No built in streaming support.
    - No async support without additional code.
    - Harder to compose with other chains.
    - More verbose nad error-prone.
    
    """

    #Step 1: Retrieve relavent documents.
    docs = retriever.invoke(query)

    # Step 2: Format documents into context strings.
    context = format_docs(docs)

    #Step 3: Fromat the prompt with context and question.
    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)

    return response.content


# Implementation 2: With LCEL - BETTER Approach.

def retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL(Lang Chain Expression Language).
    Returns a chain that can be invoked with {"question":"...."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operation with pipe operator(|)
    - Built in streaming: chain.stream() works out of the box.
    - Built in async: chain.ainvoke() and chain.astream() available.
    - Batch Processing: chain.batch() for multiple inputs.
    - Type Safety: Better intergation with Langchains type system.
    - Less code: More concise and readable.
    - Reusable: chain can be saved, shared and composed with other chains.
    - Better Debugging: Langchain provides better observability tools.
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=(
                RunnableLambda(itemgetter("question"))
                | retriever 
                | format_docs
            )
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain



if __name__ == "__main__":
    print("Retrieving...")

    query = "What is pincone in machine learning?"

    # Option 0: Raw invocation without RAG.
    print("\n"+"="*70)
    print("Implementation 0: RAW LLM Invocation (No RAG)")
    print("="*70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\n Answer")
    print(result_raw.content)

    #Option 1: User RAG implementation WITHOUT LCEL.
    print("\n"+"="*70)
    print("Implementation 1: WITHOUT LCEL")
    print("="*70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\n Answer")
    print(result_without_lcel)

    #Option2: RAG Implementation with LCEL.
    print("\n"+"="*70)
    print("Implementation 2: WITH LCEL")
    print("="*70)
    chain_with_lcel = retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)


