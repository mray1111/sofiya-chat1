import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# from dotenv import load_dotenv
import pickle




# Sidebar contents
with st.sidebar:
    st.title("Sofiya Memorial School(HS)")
    st.markdown(
        """
        ## About  
        This cool app is a support system for my students for all summative exams + weekly tests.  
        They can ask doubts about whether a topic has been covered in class, if it is important, and what to study.  
        Just ask a question, and it will tell you if it's important, what to study, and provide additional tips.
        """
    )
    add_vertical_space(5)
    st.write("Made in Sofiya Memorial School")


# Main function
def main():
    st.header("Write your questions here")
    load_dotenv()
    text = (
        "important - What is scalar quantity? "
        "Important - What is vector quantity. "
        "Not important - What is momentum. "
        "Marks of Manish is 25 out of 30. "
        "Marks of IIT Guwahati is 35 out of 35."
    )
    #st.write(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    st.write(chunks)

    #embeddings

    # embeddings=OpenAIEmbeddings()
    # VectorStore=FAISS.from_texts(chunks,embedding=embeddings)
    # store_name='documents'
    # with open(f"{store_name}.pkl","wb") as f:
    #     pickle.dump(VectorStore,f)

if __name__ == "__main__":
    main()
