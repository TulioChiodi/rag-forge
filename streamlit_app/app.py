# streamlit_app/app.py
import os

import requests
import streamlit as st

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")


def main():
    st.title("Document Q&A System")

    # Sidebar for document upload
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files", type="pdf", accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    files = [("files", file) for file in uploaded_files]
                    response = requests.post(f"{API_URL}/documents", files=files)

                    if response.status_code == 200:
                        result = response.json()
                        if result["documents_indexed"] == 0:
                            st.warning(result["message"])
                            if result["failed_files"]:
                                st.error(
                                    "Failed files: " + ", ".join(result["failed_files"])
                                )
                        else:
                            st.success(
                                f"Processed {result['documents_indexed']} documents, "
                                f"created {result['total_chunks']} chunks"
                            )
                            if result["failed_files"]:
                                st.warning(
                                    "Some files failed: "
                                    + ", ".join(result["failed_files"])
                                )
                    else:
                        st.error(f"Error: {response.text}")
        else:
            st.info("Please upload PDF documents to get started.")

    # Main area for questions
    st.header("Ask Questions")
    # Remove the need to press Enter by not storing in a variable
    st.text_input("Enter your question about the documents:", key="question")
    question = st.session_state.question  # Get the value from session state

    if st.button("Get Answer", use_container_width=True):
        if question.strip():  # Only process if question is not empty
            with st.spinner("Generating answer..."):
                response = requests.post(
                    f"{API_URL}/question", json={"question": question}
                )

                if response.status_code == 200:
                    result = response.json()
                    st.markdown("### Answer")
                    st.write(result["answer"])

                    if result["chunks"]:  # Only show context if there are chunks
                        st.markdown("### Retrieved Context")
                        for i, chunk in enumerate(result["chunks"], 1):
                            with st.expander(f"Chunk {i}"):
                                st.text(chunk)
                else:
                    st.error(f"Error: {response.text}")

    # Optional: Add a section to show system status
    with st.expander("System Status"):
        try:
            response = requests.get(f"{API_URL}/health")  # Changed to health endpoint
            if response.status_code == 200:
                st.success("API is running")
            else:
                st.warning("API is running but might have issues")
        except requests.RequestException:
            st.error("API is not running")


if __name__ == "__main__":
    main()
