"""Streamlit interface for the BERT document review assistant."""

from __future__ import annotations

import streamlit as st

from pipeline import (
    CLAUSE_DESCRIPTIONS,
    EMBEDDING_MODEL_NAME,
    NER_MODEL_NAME,
    QA_MODEL_NAME,
    BertDocumentAssistant,
    extract_text,
    normalise_text,
    split_for_qa,
    split_sentences,
)


st.set_page_config(page_title="Document Review Assistant", page_icon="DRA", layout="wide")


@st.cache_resource(show_spinner="Loading the three BERT models. This happens once per session.")
def load_assistant() -> BertDocumentAssistant:
    return BertDocumentAssistant()


def confidence_label(score: float | None) -> str:
    if score is None:
        return "Fixed-format match"
    return f"BERT confidence: {score:.1%}"


def main() -> None:
    st.title("Document Review Assistant")
    st.caption("Encoder-only workflow using BERT for Named Entity Recognition, question answering, and embeddings.")

    with st.sidebar:
        st.header("Models used")
        st.write(f"NER: `{NER_MODEL_NAME}`")
        st.write(f"Question answering: `{QA_MODEL_NAME}`")
        st.write(f"Embeddings: `{EMBEDDING_MODEL_NAME}`")
        st.info("The assistant extracts evidence from the uploaded document. It does not generate legal, HR, or compliance advice.")

    uploaded_file = st.file_uploader("Upload a business document", type=["txt", "pdf", "docx"])
    if uploaded_file is None:
        st.info("Upload a TXT, PDF, or DOCX file to begin the review.")
        return

    try:
        document_text = normalise_text(extract_text(uploaded_file.name, uploaded_file.getvalue()))
    except Exception as error:
        st.error(f"The document could not be read: {error}")
        return

    if not document_text:
        st.warning("No readable text was found. For a scanned PDF, use a text-based PDF or add OCR first.")
        return

    assistant = load_assistant()
    tabs = st.tabs(["Overview", "Entity scan", "Ask document", "Writing assistant", "Summary", "Clause review"])

    with tabs[0]:
        st.subheader("Document overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Words", len(document_text.split()))
        col2.metric("Sentences", len(split_sentences(document_text)))
        col3.metric("QA chunks", len(split_for_qa(document_text)))
        with st.expander("View extracted text"):
            st.text(document_text)

    with tabs[1]:
        st.subheader("Named Entity Recognition and structured-value scan")
        st.write("BERT identifies people, organisations, and locations. Transparent matching also finds common fixed-format values such as email addresses and phone numbers.")
        if st.button("Scan document", key="scan"):
            with st.spinner("Running the document scan..."):
                findings = assistant.scan_document(document_text)
            st.session_state["findings"] = findings

        findings = st.session_state.get("findings", [])
        if findings:
            rows = [
                {"Value": finding.text, "Category": finding.category, "Evidence": confidence_label(finding.confidence)}
                for finding in findings
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            redacted = assistant.redact_text(document_text, findings)
            with st.expander("View redacted preview"):
                st.text(redacted)
        elif "findings" in st.session_state:
            st.info("No entities or fixed-format values were detected.")

    with tabs[2]:
        st.subheader("Ask a plain-English question")
        question = st.text_input("Question", placeholder="For example: When must the invoice be paid?")
        if st.button("Find answer", key="question"):
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                with st.spinner("Searching document chunks with BERT QA..."):
                    result = assistant.answer_question(question, document_text)
                if result and result["confidence"] >= 0.05:
                    st.success(str(result["answer"]))
                    st.caption(f"Answer confidence: {result['confidence']:.1%}")
                    with st.expander("Evidence chunk"):
                        st.write(result["context"])
                else:
                    st.info("No reliable extractive answer was found. Try rephrasing the question.")

    with tabs[3]:
        st.subheader("Writing assistant for ambiguous business language")
        st.write("This tool uses base BERT embeddings to compare your sentence with clear and ambiguous business-language examples.")
        sentence = st.text_area("Sentence to assess", placeholder="For example: The supplier should respond promptly when practical.")
        if st.button("Assess wording", key="ambiguity"):
            if not sentence.strip():
                st.warning("Enter a sentence first.")
            else:
                with st.spinner("Comparing sentence embeddings..."):
                    assessment = assistant.assess_ambiguity(sentence)
                if assessment["verdict"] == "Potentially ambiguous":
                    st.warning("Potentially ambiguous: add a specific actor, action, deadline, condition, or measurable standard.")
                else:
                    st.success("More specific than the ambiguous reference examples.")
                left, right = st.columns(2)
                left.metric("Similarity to clear examples", f"{assessment['clear_score']:.3f}")
                right.metric("Similarity to ambiguous examples", f"{assessment['ambiguous_score']:.3f}")
                if assessment["cues"]:
                    st.caption("Possible vague cues: " + ", ".join(assessment["cues"]))

    with tabs[4]:
        st.subheader("Extractive document summary")
        st.write("Base BERT ranks source sentences. The result contains only sentences already present in the document.")
        sentence_count = st.slider(
            "Number of sentences to include",
            min_value=1,
            max_value=min(20, len(split_sentences(document_text))),
            value=min(4, len(split_sentences(document_text))),
        )
        if st.button("Create extractive summary", key="summary"):
            with st.spinner("Ranking source sentences..."):
                summary = assistant.extractive_summary(document_text, max_sentences=sentence_count)
            for index, sentence in enumerate(summary, start=1):
                st.write(f"{index}. {sentence}")

    with tabs[5]:
        st.subheader("Clause review")
        st.write("Base BERT embeddings compare document chunks with clause descriptions. Similarity is a review signal, not a legal conclusion.")
        with st.expander("Clause descriptions used for similarity checks", expanded=True):
            for label, description in CLAUSE_DESCRIPTIONS.items():
                st.markdown(f"- {label}: {description}")
        if st.button("Review clauses", key="clauses"):
            with st.spinner("Comparing document chunks with clause descriptions..."):
                reviews = assistant.review_clauses(document_text)
            for review in reviews:
                with st.expander(f"{review['category']} - similarity {review['similarity']:.3f}"):
                    st.caption(f"Matched against: {CLAUSE_DESCRIPTIONS[review['category']]}")
                    st.write(review["evidence"])


if __name__ == "__main__":
    main()
