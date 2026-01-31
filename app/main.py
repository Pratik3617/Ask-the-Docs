import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Ask the Docs",
    layout="centered"
)

st.title("Ask the Docs")
st.caption("Ask questions grounded strictly in your uploaded document")


# document upload section
st. header("Upload Document")

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT document",
    type=["pdf", "txt"]
)

if uploaded_file is not None:
    with st.spinner("Uploading and indexing document..."):
        files = {
            "file":(
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        try:
            response = requests.post(
                f"{API_BASE_URL}/ingest",
                files=files,
                timeout=300
            )

            if response.status_code == 200:
                st.success("Document ingested successfully")
                st.json(response.json())
            else:
                st.error("Failed to ingest document")
                st.write(response.json())
        
        except requests.exceptions.RequestException as e:
            st.error("Backend service is not reachable")
            st.write(str(e))


# Question Answering Section

st.header("Ask a Question")

question = st.text_input(
    "Enter your question about the document"
)

top_k = st.slider(
    "Number of context chunks to retrieve",
    min_value=1,
    max_value=8,
    value=4
)

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Retrieving answer..."):
            payload = {
                "question": question,
                "top_k": top_k
            }

            try:
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json=payload,
                    timeout=300
                )

                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.subheader("Answer")
                    st.write(answer)
                else:
                    st.error("Failed to get answer")
                    st.write(response.json())

            except requests.exceptions.RequestException as e:
                st.error("Backend service is not reachable")
                st.write(str(e))