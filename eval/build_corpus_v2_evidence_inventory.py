from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
MANIFEST_PATH = EVAL_DIR / "corpus_v2_manifest.json"
AUDIT_PATH = EVAL_DIR / "corpus_v2_ingestion_audit_latest.json"
CATALOG_PATH = EVAL_DIR / "corpus_v2_evidence_catalog.json"
HARD_NEGATIVE_PATH = EVAL_DIR / "corpus_v2_hard_negative_map.json"
PERSIST_DIRECTORY = EVAL_DIR / ".chroma_corpus_v2_audit"
COLLECTION_NAME = "documind_corpus_v2_ingestion_audit"


def _item(
    evidence_id: str,
    document_id: str,
    chunk_index: int,
    topic: str,
    claim: str,
    phrases: list[str],
    *,
    evidence_type: str = "single_chunk",
    notes: str = "",
    ambiguity: str | None = None,
    supporting_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "document_id": document_id,
        "canonical_chunk_index": chunk_index,
        "topic": topic,
        "concise_factual_claim": claim,
        "evidence_keywords_or_phrases": phrases,
        "evidence_type": evidence_type,
        "notes": notes,
        "ambiguity": ambiguity,
        "supporting_chunks": supporting_chunks or [],
    }


EVIDENCE_SPECS = [
    _item("ev-study-001", "corpus-v2-study-plan", 0, "programme_duration", "The MSc Computer Science indicative study duration is 24 months.", ["Indicative Study Duration", "24 Months"]),
    _item("ev-study-002", "corpus-v2-study-plan", 0, "programme_credits", "The programme carries 180 credits.", ["Credits", "180"]),
    _item("ev-study-003", "corpus-v2-study-plan", 0, "module_schedule", "Global Trends in Computer Science is a 15-credit, eight-week module.", ["Global Trends in Computer Science", "15", "8 weeks"]),
    _item("ev-study-004", "corpus-v2-study-plan", 0, "elective_choice", "Machine Learning in Practice, Reasoning and Intelligent Systems, and Robotics appear in one elective choice group.", ["Machine Learning in Practice", "Reasoning and Intelligent Systems", "Robotics"]),
    _item("ev-study-005", "corpus-v2-study-plan", 1, "capstone", "The final Computer Science Capstone Project is worth 60 credits and lasts 32 weeks.", ["Final Module", "Computer Science Capstone Project", "60", "32 weeks"]),
    _item("ev-study-006", "corpus-v2-study-plan", 2, "study_breaks", "A study break or reassessment can prevent completion within the indicative 24 months.", ["break from your studies", "reassessment", "24 months"]),
    _item("ev-study-007", "corpus-v2-study-plan", 0, "programme_time_limits", "The indicative duration is 24 months, while the maximum completion period is 72 months.", ["Indicative Study Duration", "24 Months"], evidence_type="multi_chunk_candidate", supporting_chunks=[{"chunk_index": 2, "phrases": ["maximum period", "72 months", "6 years"]}], notes="Requires the headline duration and the separate maximum-period note."),

    _item("ev-admissions-001", "corpus-v2-admissions-appeals", 10, "appeal_definition", "An admissions appeal is a formal request to reconsider an application decision.", ["appeal is a formal request", "reconsideration of a decision"]),
    _item("ev-admissions-002", "corpus-v2-admissions-appeals", 10, "complaint_definition", "A complaint expresses dissatisfaction with admissions procedures, implementation, actions, or inaction.", ["complaint is defined", "expression of dissatisfaction", "admissions procedures"]),
    _item("ev-admissions-003", "corpus-v2-admissions-appeals", 10, "appeal_scope", "The procedure cannot be used merely to dispute admissions staff's academic or professional judgement.", ["may not make an Appeal or Complaint", "academic or professional judgement"]),
    _item("ev-admissions-004", "corpus-v2-admissions-appeals", 5, "representation", "A third party may submit an appeal or complaint only with the applicant's explicit written consent.", ["third party", "explicit consent", "in writing"]),
    _item("ev-admissions-005", "corpus-v2-admissions-appeals", 5, "anonymous_complaints", "Anonymous complaints are not handled under this procedure.", ["Anonymous complaints", "will not be dealt with"]),
    _item("ev-admissions-006", "corpus-v2-admissions-appeals", 12, "appeal_deadline", "Applicants should normally appeal within one month of the application decision.", ["make an appeal within one month", "receiving the decision"]),
    _item("ev-admissions-007", "corpus-v2-admissions-appeals", 15, "response_time", "The formal appeal review normally aims for a response within fifteen working days after receipt and acknowledgement.", ["fifteen working days", "receipt and acknowledgement"]),
    _item("ev-admissions-008", "corpus-v2-admissions-appeals", 21, "independent_investigation", "The Executive Pro-Vice-Chancellor acknowledges an escalated complaint within ten working days and appoints investigators with no material interest.", ["within ten working days", "no material interest", "carry out an investigation"]),
    _item("ev-admissions-009", "corpus-v2-admissions-appeals", 12, "appeal_timeline", "A normal appeal should be made within one month, while the later Executive Pro-Vice-Chancellor stage has a ten-working-day acknowledgement target.", ["appeal within one month", "receiving the decision"], evidence_type="multi_chunk_candidate", supporting_chunks=[{"chunk_index": 21, "phrases": ["within ten working days", "acknowledge receipt"]}], notes="Combines deadlines from different stages of the procedure."),

    _item("ev-integrity-001", "corpus-v2-appendix-l-cop-assess", 4, "integrity_categories", "The policy distinguishes four academic-integrity categories with increasing levels of dishonesty.", ["Category 1 is determined", "Category 2 captures", "Category 4 covers"]),
    _item("ev-integrity-002", "corpus-v2-appendix-l-cop-assess", 5, "graduation_restriction", "A student with an unresolved academic-integrity matter cannot graduate, including while an appeal is registered.", ["No student with an unresolved academic integrity matter", "graduate", "registered an appeal"]),
    _item("ev-integrity-003", "corpus-v2-appendix-l-cop-assess", 5, "extenuating_circumstances", "Personal Extenuating Circumstances are not accepted as a reason for committing academic misconduct.", ["Personal Extenuating Circumstances", "will not be accepted", "academic misconduct"]),
    _item("ev-integrity-004", "corpus-v2-appendix-l-cop-assess", 12, "poor_academic_practice", "Poor academic practice can include poor paraphrasing or inadequate referencing.", ["Poor academic practice", "poor paraphrasing", "inadequate referencing"]),
    _item("ev-integrity-005", "corpus-v2-appendix-l-cop-assess", 13, "collusion", "Collusion involves conscious collaboration on work later submitted as individual work without official approval.", ["official approval", "consciously collaborate", "individual efforts"]),
    _item("ev-integrity-006", "corpus-v2-appendix-l-cop-assess", 14, "commissioned_work", "Commissioned or procured work includes asking a third party to prepare all or part of an assignment, paid or unpaid.", ["Commissioned or Procured", "third parties", "with or without payment"]),
    _item("ev-integrity-007", "corpus-v2-appendix-l-cop-assess", 15, "proofreading", "Acceptable proofreading identifies presentation errors; adding, rewriting, or fact-checking content is unacceptable.", ["Acceptable proof-reading", "Unacceptable proof-reading", "rewriting any content"]),
    _item("ev-integrity-008", "corpus-v2-appendix-l-cop-assess", 16, "generative_ai_use", "Submitting GAI-generated assessment work contrary to assignment guidance is treated as dishonest practice.", ["Generative Artificial Intelligence", "contrary to the guidance", "assignment brief"]),
    _item("ev-integrity-009", "corpus-v2-appendix-l-cop-assess", 54, "integrity_appeal", "Category 2–4 decisions may be appealed only on procedural-error grounds in the investigation.", ["Right of Appeal", "Category 2, 3 or 4", "procedural error"]),
    _item("ev-integrity-010", "corpus-v2-appendix-l-cop-assess", 4, "category_and_appeal_rules", "The policy defines escalating integrity categories, while appeals against Category 2–4 decisions are limited to procedural-error grounds.", ["Category 2 captures", "Category 4 covers"], evidence_type="multi_chunk_candidate", supporting_chunks=[{"chunk_index": 54, "phrases": ["Category 2, 3 or 4", "procedural error"]}], notes="The category definitions and appeal restriction are separated by several pages."),

    _item("ev-coding-001", "corpus-v2-coding-standards", 0, "coding_standard_purpose", "The coding standard aims to make assignment and project code more readable and easier to follow.", ["assignments and projects", "more readable", "easier"]),
    _item("ev-coding-002", "corpus-v2-coding-standards", 1, "variable_naming", "Variable and attribute identifiers should be meaningful, start lowercase, and use camel case.", ["Variables/Attributes", "meaningful", "lower case", "camel case"]),
    _item("ev-coding-003", "corpus-v2-coding-standards", 1, "class_naming", "Class identifiers start uppercase and should be nouns.", ["Class identifiers", "uppercase letter", "noun"]),
    _item("ev-coding-004", "corpus-v2-coding-standards", 2, "constant_naming", "Constant identifiers are normally uppercase with underscores between words.", ["Identifiers of constants", "uppercase", "underscores"]),
    _item("ev-coding-005", "corpus-v2-coding-standards", 3, "function_naming", "Method names start lowercase, use camel case, and should be verbs.", ["Method / Function Names", "camel case", "verbs"]),
    _item("ev-coding-006", "corpus-v2-coding-standards", 3, "code_comments", "Comments should explain a code section's intention and outline its algorithm.", ["explain what the intention", "outline of the algorithm"]),
    _item("ev-coding-007", "corpus-v2-coding-standards", 5, "public_interface_documentation", "Public methods should be commented, including input variables and returned values.", ["public method", "input variables", "returned values"]),
    _item("ev-coding-008", "corpus-v2-coding-standards", 6, "source_attribution", "Online or textbook code ideas and fragments must be precisely referenced in comments.", ["On-line Resources and Textbooks", "include references", "comments"]),
    _item("ev-coding-009", "corpus-v2-coding-standards", 1, "identifier_conventions", "Variables use lowercase-leading camel case, classes begin uppercase, and constants are normally uppercase with underscores.", ["Variables/Attributes", "camel case", "Class identifiers"], evidence_type="multi_chunk_candidate", supporting_chunks=[{"chunk_index": 2, "phrases": ["Identifiers of constants", "uppercase", "underscores"]}], notes="Combines separate identifier rules rather than treating one naming rule as sufficient."),

    _item("ev-ethics-001", "corpus-v2-ai-ethics-chapter-1", 6, "computing_history", "The Apollo Guidance Computer is described as having 32,768 bits of RAM and 72 KB of ROM.", ["Apollo Guidance Computer", "32,768 bits", "72 KB"]),
    _item("ev-ethics-002", "corpus-v2-ai-ethics-chapter-1", 8, "internet_use", "A cited 2018 UK estimate reports 25.3 hours of internet use per week, 15.4 hours more than in 2005.", ["25.3 hours per week", "15.4 hours", "2005"]),
    _item("ev-ethics-003", "corpus-v2-ai-ethics-chapter-1", 12, "ai_methods", "The chapter characterises modern AI's main approach as statistical inference and correlation rather than logical deduction.", ["no longer logical deduction", "statistical inference and correlation"]),
    _item("ev-ethics-004", "corpus-v2-ai-ethics-chapter-1", 18, "data_identity", "The chapter describes self-identity and personal data as increasingly coupled.", ["Self-identity and personal data", "glued together"]),
    _item("ev-ethics-005", "corpus-v2-ai-ethics-chapter-1", 25, "right_to_be_forgotten", "Removing links in only one search-engine jurisdiction may not effectively implement the right to be forgotten.", ["right to be forgotten", "remove links", "all versions"], ambiguity="This is a jurisdiction-specific example in the chapter; do not generalize it into a universal legal rule."),
    _item("ev-ethics-006", "corpus-v2-ai-ethics-chapter-1", 33, "digital_reengineering", "Re-ontologizing means re-engineering that transforms a system's intrinsic nature, not merely its structure.", ["re-ontologizing", "intrinsic nature", "ontology"]),
    _item("ev-ethics-007", "corpus-v2-ai-ethics-chapter-1", 37, "digital_agency", "The chapter connects digitally transformed political agency with direct democracy and artificial agency with AI.", ["political agency as direct democracy", "artificial agency as AI"]),
    _item("ev-ethics-008", "corpus-v2-ai-ethics-chapter-1", 56, "responsible_design", "The chapter argues for information societies that are open, tolerant, equitable, just, environmentally supportive, and respectful of human dignity.", ["open, tolerant, equitable, just", "human dignity", "better design"]),

    _item("ev-terms-001", "corpus-v2-uolo-programme-terms", 4, "cancellation_period", "The Cancellation Period runs until 21 days from a programme's Start Date.", ["Cancellation Period", "21 days", "Start Date"]),
    _item("ev-terms-002", "corpus-v2-uolo-programme-terms", 14, "cancellation_refund", "During the Cancellation Period a student may end the contract without Net Tuition Fees, or receive a full refund if already paid.", ["end this contract", "full refund", "Cancellation Period"]),
    _item("ev-terms-003", "corpus-v2-uolo-programme-terms", 19, "programme_changes", "Students are normally consulted in advance about significant programme changes, subject to regulatory, legal, or external-event exceptions.", ["significant change", "consulted with in advance", "regulatory or legal reasons"]),
    _item("ev-terms-004", "corpus-v2-uolo-programme-terms", 22, "maximum_registration", "There is no automatic right to study beyond the Maximum Period of Registration, though an extension may be granted at the University's discretion.", ["no automatic right", "Maximum Period of Registration", "grant an extension"]),
    _item("ev-terms-005", "corpus-v2-uolo-programme-terms", 32, "fees_and_duration", "Taking longer than the indicative duration does not add fees beyond the stated Net Tuition Fees, subject to specified increases.", ["longer than the Indicative Study Duration", "no additional fees", "Net Tuition Fees"]),
    _item("ev-terms-006", "corpus-v2-uolo-programme-terms", 35, "fee_default", "Kaplan gives five working days' written notice to settle an outstanding account balance.", ["five (5) working days", "settle your outstanding account balance"]),
    _item("ev-terms-007", "corpus-v2-uolo-programme-terms", 58, "student_intellectual_property", "Students generally own IP created during the programme, subject to listed exceptions.", ["you will own all Intellectual Property", "subject to the exceptions"]),
    _item("ev-terms-008", "corpus-v2-uolo-programme-terms", 64, "programme_complaints", "Kaplan is the initial contact for complaints relating to an online programme.", ["Kaplan will be your initial point of contact", "complaint relating to your Programme"]),
    _item("ev-terms-009", "corpus-v2-uolo-programme-terms", 92, "governing_law", "The programme terms are governed by English law and submit to the English courts' jurisdiction.", ["governed by English law", "exclusive jurisdiction of the English courts"]),
    _item("ev-terms-010", "corpus-v2-uolo-programme-terms", 54, "personal_data", "The University and Kaplan collect and process personal data, including listed sensitive categories, for programme-related responsibilities.", ["collect, retain and process", "sensitive personal data", "education and support obligations"]),
    _item("ev-terms-011", "corpus-v2-uolo-programme-terms", 22, "account_security", "Students must not make their online passwords available to third parties.", ["online passwords", "third parties"]),
    _item("ev-terms-012", "corpus-v2-uolo-programme-terms", 4, "cancellation_window_and_effect", "The cancellation window lasts 21 days from the Start Date and permits ending the contract without Net Tuition Fees or with a full refund if already paid.", ["Cancellation Period", "21 days", "Start Date"], evidence_type="multi_chunk_candidate", supporting_chunks=[{"chunk_index": 14, "phrases": ["end this contract", "full refund", "Cancellation Period"]}], notes="Definition and financial consequence occur on different pages."),
    _item("ev-terms-013", "corpus-v2-uolo-programme-terms", 66, "online_study_it_requirements", "Online study requires specified equipment and connectivity, including a computer, headset, broadband, current browser, and office software.", ["IT and information security", "PC or laptop", "broadband connection", "up-to-date web browser"]),

    _item("ev-serverless-001", "corpus-v2-serverless-computing", 3, "faas", "FaaS lets developers run functions without building and managing the underlying infrastructure.", ["FaaS", "without incurring the complexity", "underlying infrastructure"]),
    _item("ev-serverless-002", "corpus-v2-serverless-computing", 24, "serverless_workflow", "The serverless workflow is divided into function programming and function serving.", ["two phases", "function programming", "function serving"]),
    _item("ev-serverless-003", "corpus-v2-serverless-computing", 28, "hostless_operations", "The hostless model reduces server-management and patching work but requires application-level resource and execution monitoring.", ["do not need to upgrade the servers", "security patches", "resource occupation", "execution time"]),
    _item("ev-serverless-004", "corpus-v2-serverless-computing", 30, "statelessness", "Serverless functions commonly run in short-lived stateless containers without persisting ephemeral state.", ["stateless container", "short period", "ephemeral state"]),
    _item("ev-serverless-005", "corpus-v2-serverless-computing", 38, "startup_latency", "Startup latency is the period from function invocation to execution and includes warm and cold starts.", ["Startup latency", "warm start", "cold start", "period from a function"]),
    _item("ev-serverless-006", "corpus-v2-serverless-computing", 39, "cold_start", "Cold-start cost is generally significantly higher than warm-start cost.", ["cost of the cold start", "higher than that of the warm start"]),
    _item("ev-serverless-007", "corpus-v2-serverless-computing", 52, "isolation_security", "The paper describes containers as having weaker isolation because they rely on kernel security mechanisms.", ["container has weak isolation", "kernel security mechanism"]),
    _item("ev-serverless-008", "corpus-v2-serverless-computing", 72, "faas_baas", "The paper presents a serverless platform as a combination of FaaS and BaaS, with BaaS supporting infrastructure and auto-scaling policy.", ["FaaS", "BaaS", "auto-scaling policy"], ambiguity="The extracted phrase 'combination' is broken by PDF line hyphenation, so shorter identifying phrases are used."),
    _item("ev-serverless-009", "corpus-v2-serverless-computing", 124, "serverless_benchmarking", "Serverless benchmarks help deployments and framework selection and can measure memory, CPU, and language effects.", ["serverless benchmarking", "selection of the best suitable framework", "memory, CPU, and language"]),
    _item("ev-serverless-010", "corpus-v2-serverless-computing", 38, "cold_start_context", "Startup latency spans invocation to execution, and cold starts generally cost more than warm starts.", ["Startup latency", "warm start", "cold start"], evidence_type="multi_chunk_candidate", supporting_chunks=[{"chunk_index": 39, "phrases": ["cost of the cold start", "higher than that of the warm start"]}], notes="The definition and comparative cost are split across adjacent overlapping chunks."),
]


HARD_NEGATIVE_SPECS = [
    {"overlap_id": "hn-001", "topic": "programme duration and time limits", "evidence_ids": ["ev-study-001", "ev-study-007", "ev-terms-004", "ev-terms-005"], "why_they_could_compete": "Both sources discuss programme duration, maximum registration, breaks, and time-related consequences, but only the Study Plan supplies the MSc's exact 24/72-month figures.", "overlap_type": "both", "difficulty": "high"},
    {"overlap_id": "hn-002", "topic": "admissions appeals and programme complaints", "evidence_ids": ["ev-admissions-001", "ev-admissions-006", "ev-terms-008"], "why_they_could_compete": "The policy defines applicant appeals and deadlines, while the terms direct enrolled programme complaints to Kaplan; complaint and programme language overlaps strongly.", "overlap_type": "both", "difficulty": "high"},
    {"overlap_id": "hn-003", "topic": "appeal rights and procedural grounds", "evidence_ids": ["ev-admissions-003", "ev-integrity-009"], "why_they_could_compete": "Both restrict appeals but concern different decisions: admissions judgement versus academic-integrity procedure.", "overlap_type": "both", "difficulty": "high"},
    {"overlap_id": "hn-004", "topic": "assessment and programme rules", "evidence_ids": ["ev-study-003", "ev-integrity-001", "ev-terms-004"], "why_they_could_compete": "Study scheduling, assessment governance, and registration rules share academic-programme terminology without supporting the same facts.", "overlap_type": "semantic", "difficulty": "medium"},
    {"overlap_id": "hn-005", "topic": "AI use and responsibility", "evidence_ids": ["ev-integrity-008", "ev-ethics-003", "ev-ethics-008"], "why_they_could_compete": "One source regulates student use of generative AI; the other discusses AI methods and responsible societal design.", "overlap_type": "both", "difficulty": "high"},
    {"overlap_id": "hn-006", "topic": "software functions and coding practice", "evidence_ids": ["ev-coding-005", "ev-coding-006", "ev-serverless-001", "ev-serverless-002"], "why_they_could_compete": "Both documents repeatedly use function, code, programming, and developer vocabulary, but one specifies style while the other explains cloud execution.", "overlap_type": "both", "difficulty": "high"},
    {"overlap_id": "hn-007", "topic": "source attribution and intellectual property", "evidence_ids": ["ev-coding-008", "ev-integrity-004", "ev-terms-007"], "why_they_could_compete": "Referencing code sources, academic attribution, and ownership of student-created IP are related but legally and academically distinct.", "overlap_type": "semantic", "difficulty": "medium"},
    {"overlap_id": "hn-008", "topic": "information and infrastructure security", "evidence_ids": ["ev-terms-011", "ev-terms-013", "ev-serverless-003", "ev-serverless-007"], "why_they_could_compete": "The programme terms and serverless paper both contain IT/security vocabulary, while only the technical paper explains patching and isolation.", "overlap_type": "both", "difficulty": "medium"},
    {"overlap_id": "hn-009", "topic": "personal data, identity, and institutional processing", "evidence_ids": ["ev-ethics-004", "ev-ethics-005", "ev-terms-010"], "why_they_could_compete": "The ethics chapter discusses identity and data conceptually, whereas the terms describe institutional processing obligations; semantic proximity can mislead embeddings.", "overlap_type": "semantic", "difficulty": "medium"},
]


COVERAGE_PROPOSAL = {
    "target_case_count": 54,
    "status": "proposal_only_no_questions_generated",
    "categories": {
        "direct_factual": 10,
        "semantic_paraphrase": 10,
        "keyword_exact_term": 8,
        "contextual": 9,
        "cross_document_hard_negative": 11,
        "multi_evidence": 6,
    },
    "by_document_and_category": {
        "corpus-v2-study-plan": {"direct_factual": 2, "semantic_paraphrase": 1, "keyword_exact_term": 1, "contextual": 1, "cross_document_hard_negative": 1, "multi_evidence": 1},
        "corpus-v2-admissions-appeals": {"direct_factual": 2, "semantic_paraphrase": 1, "keyword_exact_term": 1, "contextual": 1, "cross_document_hard_negative": 2, "multi_evidence": 1},
        "corpus-v2-appendix-l-cop-assess": {"direct_factual": 1, "semantic_paraphrase": 2, "keyword_exact_term": 1, "contextual": 1, "cross_document_hard_negative": 2, "multi_evidence": 1},
        "corpus-v2-coding-standards": {"direct_factual": 1, "semantic_paraphrase": 1, "keyword_exact_term": 2, "contextual": 1, "cross_document_hard_negative": 1, "multi_evidence": 1},
        "corpus-v2-ai-ethics-chapter-1": {"direct_factual": 1, "semantic_paraphrase": 2, "keyword_exact_term": 1, "contextual": 2, "cross_document_hard_negative": 2, "multi_evidence": 0},
        "corpus-v2-uolo-programme-terms": {"direct_factual": 2, "semantic_paraphrase": 1, "keyword_exact_term": 1, "contextual": 1, "cross_document_hard_negative": 2, "multi_evidence": 1},
        "corpus-v2-serverless-computing": {"direct_factual": 1, "semantic_paraphrase": 2, "keyword_exact_term": 1, "contextual": 2, "cross_document_hard_negative": 1, "multi_evidence": 1},
    },
    "version_sensitive_future_cases": {
        "supported": True,
        "current_limit": "The manifest currently contains one version per document, so true cross-version retrieval cannot yet be scored.",
        "candidates": ["Academic Integrity Policy 2025-26", "UoLO Programme Terms updated March 2026", "Admissions policy dated October 2022"],
    },
}


def normalize_text(value: str) -> str:
    value = value.casefold().replace("\u00ad", "")
    return " ".join(re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).split())


def ensure_backend_import_path() -> None:
    backend = PROJECT_ROOT / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_stored_chunks() -> tuple[dict[tuple[str, int], dict[str, Any]], int]:
    ensure_backend_import_path()
    from app.config.settings import AppSettings
    from app.services.chroma_vector_store import ChromaPersistentVectorStore
    from chromadb.api.shared_system_client import SharedSystemClient

    settings = AppSettings.from_env()
    store = ChromaPersistentVectorStore(
        persist_directory=str(PERSIST_DIRECTORY),
        collection_name=COLLECTION_NAME,
        embedding_model_name=settings.embedding_model_name,
        embedding_score_weight=settings.embedding_score_weight,
        bm25_score_weight=settings.bm25_score_weight,
    )
    try:
        records = store.collection.get(include=["documents", "metadatas"])
        chunks: dict[tuple[str, int], dict[str, Any]] = {}
        for record_id, text, metadata in zip(
            records.get("ids") or [],
            records.get("documents") or [],
            records.get("metadatas") or [],
        ):
            stored = metadata or {}
            key = (str(stored.get("document_id")), int(stored["chunk_index"]))
            if key in chunks:
                raise ValueError(f"Duplicate stored chunk identity: {key}")
            chunks[key] = {
                "record_id": record_id,
                "text": text or "",
                "metadata": stored,
            }
        return chunks, store.count()
    finally:
        store.client._system.stop()
        SharedSystemClient.clear_system_cache()


def phrases_present(text: str, phrases: list[str]) -> bool:
    normalized = normalize_text(text)
    return all(normalize_text(phrase) in normalized for phrase in phrases)


def chunk_ref(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = chunk["metadata"]
    return {
        "record_id": chunk["record_id"],
        "document_id": metadata["document_id"],
        "page_number": metadata["page_number"],
        "chunk_index": metadata["chunk_index"],
    }


def validate_coverage_proposal() -> None:
    category_totals = Counter()
    document_total = 0
    for allocation in COVERAGE_PROPOSAL["by_document_and_category"].values():
        category_totals.update(allocation)
        document_total += sum(allocation.values())
    if document_total != COVERAGE_PROPOSAL["target_case_count"]:
        raise ValueError("Coverage proposal document totals do not match target.")
    if dict(category_totals) != COVERAGE_PROPOSAL["categories"]:
        raise ValueError("Coverage proposal category totals do not match target.")


def build_inventory() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_json(MANIFEST_PATH)
    audit = load_json(AUDIT_PATH)
    chunks, stored_count = load_stored_chunks()
    if stored_count != audit["summary"]["reopened_stored_chunk_count"]:
        raise ValueError("Stored chunk count differs from the ingestion audit.")

    text_documents = {
        document["document_id"]: document
        for document in manifest["documents"]
        if document["include_in_text_retrieval"]
    }
    actual_document_ids = {document_id for document_id, _ in chunks}
    if actual_document_ids != set(text_documents):
        raise ValueError("Chroma document IDs differ from the text manifest.")

    evidence_ids: set[str] = set()
    evidence_items: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    for spec in EVIDENCE_SPECS:
        evidence_id = spec["evidence_id"]
        if evidence_id in evidence_ids:
            raise ValueError(f"Duplicate evidence_id: {evidence_id}")
        evidence_ids.add(evidence_id)
        document_id = spec["document_id"]
        canonical_key = (document_id, spec["canonical_chunk_index"])
        canonical = chunks.get(canonical_key)
        if canonical is None:
            validation_errors.append(f"{evidence_id}: missing canonical chunk {canonical_key}")
            continue
        if not phrases_present(canonical["text"], spec["evidence_keywords_or_phrases"]):
            validation_errors.append(f"{evidence_id}: canonical phrases not found")

        acceptable = [
            chunk_ref(chunk)
            for (candidate_document_id, _), chunk in chunks.items()
            if candidate_document_id == document_id
            and phrases_present(chunk["text"], spec["evidence_keywords_or_phrases"])
        ]
        supporting_refs: list[dict[str, Any]] = []
        for support in spec["supporting_chunks"]:
            support_key = (document_id, support["chunk_index"])
            support_chunk = chunks.get(support_key)
            if support_chunk is None:
                validation_errors.append(f"{evidence_id}: missing supporting chunk {support_key}")
                continue
            if not phrases_present(support_chunk["text"], support["phrases"]):
                validation_errors.append(f"{evidence_id}: supporting phrases not found in {support_key}")
            supporting_refs.append({**chunk_ref(support_chunk), "identifying_phrases": support["phrases"]})

        metadata = canonical["metadata"]
        manifest_document = text_documents[document_id]
        if metadata.get("file_name") != manifest_document["file_name"]:
            validation_errors.append(f"{evidence_id}: Chroma/manifest file_name mismatch")
        evidence_items.append({
            "evidence_id": evidence_id,
            "document_id": document_id,
            "file_name": metadata["file_name"],
            "page_number": metadata["page_number"],
            "chunk_index": metadata["chunk_index"],
            "topic": spec["topic"],
            "concise_factual_claim": spec["concise_factual_claim"],
            "evidence_keywords_or_phrases": spec["evidence_keywords_or_phrases"],
            "evidence_type": spec["evidence_type"],
            "acceptable_chunk_refs": sorted(acceptable, key=lambda ref: ref["chunk_index"]),
            "supporting_chunk_refs": supporting_refs,
            "ambiguity": spec["ambiguity"],
            "notes": spec["notes"],
        })

    if validation_errors:
        raise ValueError("Evidence validation failed:\n- " + "\n- ".join(validation_errors))

    evidence_by_id = {item["evidence_id"]: item for item in evidence_items}
    hard_negative_relationships: list[dict[str, Any]] = []
    relationships_by_document = Counter()
    for spec in HARD_NEGATIVE_SPECS:
        referenced_items = [evidence_by_id[evidence_id] for evidence_id in spec["evidence_ids"]]
        relevant_documents = sorted({item["document_id"] for item in referenced_items})
        if len(relevant_documents) < 2:
            raise ValueError(f"Hard negative {spec['overlap_id']} is not cross-document.")
        relationships_by_document.update(relevant_documents)
        hard_negative_relationships.append({
            **spec,
            "relevant_documents": relevant_documents,
            "representative_chunk_identities": [
                {
                    "evidence_id": item["evidence_id"],
                    "document_id": item["document_id"],
                    "page_number": item["page_number"],
                    "chunk_index": item["chunk_index"],
                }
                for item in referenced_items
            ],
        })

    distribution: dict[str, Any] = {}
    for document_id, document in text_documents.items():
        document_items = [item for item in evidence_items if item["document_id"] == document_id]
        pages = {item["page_number"] for item in document_items}
        represented_chunks = {
            (item["page_number"], item["chunk_index"])
            for item in document_items
        }
        for item in document_items:
            pages.update(
                ref["page_number"] for ref in item["supporting_chunk_refs"]
            )
            represented_chunks.update(
                (ref["page_number"], ref["chunk_index"])
                for ref in item["supporting_chunk_refs"]
            )
        distribution[document_id] = {
            "display_name": document["display_name"],
            "useful_evidence_item_count": len(document_items),
            "pages_represented": sorted(pages),
            "page_count_represented": len(pages),
            "chunks_represented": [
                {"page_number": page, "chunk_index": chunk}
                for page, chunk in sorted(represented_chunks)
            ],
            "chunk_count_represented": len(represented_chunks),
            "major_topics": sorted({item["topic"] for item in document_items}),
            "potential_hard_negative_relationship_count": relationships_by_document[document_id],
        }

    ambiguous_items = [item["evidence_id"] for item in evidence_items if item["ambiguity"]]
    multi_candidates = [item["evidence_id"] for item in evidence_items if item["evidence_type"] == "multi_chunk_candidate"]
    validate_coverage_proposal()
    catalog = {
        "catalog_version": "1",
        "corpus_id": manifest["corpus_id"],
        "source_manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_collection": COLLECTION_NAME,
        "stored_chunk_count": stored_count,
        "summary": {
            "evidence_item_count": len(evidence_items),
            "text_document_count": len(text_documents),
            "ambiguous_item_count": len(ambiguous_items),
            "multi_chunk_candidate_count": len(multi_candidates),
            "validation_passed": True,
        },
        "evidence_distribution": distribution,
        "ambiguous_evidence_items": ambiguous_items,
        "excluded_ambiguous_candidates": [
            {
                "source": "Serverless Computing figures",
                "reason": "Diagram-only relationships were not catalogued because the current text pipeline retrieves captions and nearby prose, not the visual structure itself.",
            },
            {
                "source": "Study Plan second elective group",
                "reason": "The full list crosses the c0/c1 character boundary; individual complete facts remain usable, but the split list was not forced into a single-chunk ground truth.",
            },
            {
                "source": "PDF headers, download notices, and reference lists",
                "reason": "Layout artefacts and bibliographic mentions were excluded because they do not provide useful benchmark claims.",
            },
        ],
        "multi_chunk_candidates": multi_candidates,
        "evidence_items": evidence_items,
        "evaluation_coverage_proposal": COVERAGE_PROPOSAL,
    }
    hard_negative_map = {
        "map_version": "1",
        "corpus_id": manifest["corpus_id"],
        "summary": {
            "overlap_topic_count": len(hard_negative_relationships),
            "high_difficulty_count": sum(item["difficulty"] == "high" for item in hard_negative_relationships),
            "medium_difficulty_count": sum(item["difficulty"] == "medium" for item in hard_negative_relationships),
        },
        "overlaps": hard_negative_relationships,
    }
    return catalog, hard_negative_map


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> int:
    catalog, hard_negative_map = build_inventory()
    write_json(CATALOG_PATH, catalog)
    write_json(HARD_NEGATIVE_PATH, hard_negative_map)
    print("Corpus V2 evidence inventory complete")
    print(f"Evidence items: {catalog['summary']['evidence_item_count']}")
    print(f"Documents: {catalog['summary']['text_document_count']}")
    print(f"Multi-chunk candidates: {catalog['summary']['multi_chunk_candidate_count']}")
    print(f"Hard-negative overlap topics: {hard_negative_map['summary']['overlap_topic_count']}")
    print(f"Catalog written to: {CATALOG_PATH}")
    print(f"Hard-negative map written to: {HARD_NEGATIVE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
