from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent
CATALOG_PATH = EVAL_DIR / "corpus_v2_evidence_catalog.json"
CASES_PATH = EVAL_DIR / "corpus_v2_eval_cases.json"


def grounded(
    slug: str,
    category: str,
    question: str,
    evidence_id: str,
    *,
    hard_negative_overlap_id: str | None = None,
) -> dict[str, Any]:
    return {
        "slug": slug,
        "category": category,
        "question": question,
        "evidence_ids": [evidence_id],
        "hard_negative_overlap_id": hard_negative_overlap_id,
    }


GROUNDED_SPECS = [
    # Study Plan: 7
    grounded("study_duration", "direct_factual", "What is the indicative duration of the MSc Computer Science programme?", "ev-study-001"),
    grounded("study_credits", "direct_factual", "How many credits does the MSc Computer Science programme carry?", "ev-study-002"),
    grounded("study_break_effect", "semantic_paraphrase", "What circumstances can make the usual two-year completion schedule unachievable?", "ev-study-006"),
    grounded("capstone_exact", "keyword_exact_term", "What credit value and duration are listed for the Computer Science Capstone Project?", "ev-study-005"),
    grounded("elective_group_context", "contextual", "If a student is choosing among Machine Learning in Practice, Reasoning and Intelligent Systems, and Robotics, what kind of choice is being made?", "ev-study-004"),
    grounded("study_duration_competition", "cross_document_hard_negative", "For the MSc Computer Science study plan specifically, what indicative duration is printed rather than the general maximum-registration rule?", "ev-study-001", hard_negative_overlap_id="hn-001"),
    grounded("study_duration_and_maximum", "multi_evidence", "What are both the indicative study duration and the maximum permitted completion period for this MSc?", "ev-study-007"),

    # Admissions: 8
    grounded("appeal_definition", "direct_factual", "How does the admissions procedure define an appeal?", "ev-admissions-001"),
    grounded("appeal_deadline", "direct_factual", "How soon should an applicant normally appeal after receiving an application decision?", "ev-admissions-006"),
    grounded("anonymous_complaint_paraphrase", "semantic_paraphrase", "Will the applicant procedure investigate a grievance when the person raising it does not identify themselves?", "ev-admissions-005"),
    grounded("written_consent_exact", "keyword_exact_term", "What explicit consent is required before a third party may submit an Admissions Appeal or Complaint for an applicant?", "ev-admissions-004"),
    grounded("judgement_scope", "contextual", "Can an unsuccessful applicant use this procedure simply because they disagree with admissions staff's assessment of their suitability?", "ev-admissions-003"),
    grounded("applicant_vs_student_complaint", "cross_document_hard_negative", "An applicant wants reconsideration of an application decision. Is this an admissions appeal or an enrolled student's programme complaint, and how is it defined?", "ev-admissions-001", hard_negative_overlap_id="hn-002"),
    grounded("admissions_judgement_vs_integrity", "cross_document_hard_negative", "Which restriction applies when an applicant tries to challenge admissions staff's academic or professional judgement?", "ev-admissions-003", hard_negative_overlap_id="hn-003"),
    grounded("appeal_stage_timelines", "multi_evidence", "What are the normal initial appeal deadline and the acknowledgement target at the later Executive Pro-Vice-Chancellor complaint stage?", "ev-admissions-009"),

    # Appendix L: 8
    grounded("integrity_graduation", "direct_factual", "May a student graduate while an academic-integrity matter remains unresolved?", "ev-integrity-002"),
    grounded("pecs_misconduct", "semantic_paraphrase", "Can personal mitigating circumstances justify committing academic misconduct under the policy?", "ev-integrity-003"),
    grounded("commissioned_work_paraphrase", "semantic_paraphrase", "How does the policy treat having another person prepare some or all of an assignment, whether paid or unpaid?", "ev-integrity-006"),
    grounded("collusion_exact", "keyword_exact_term", "Under Appendix L, what conduct is described as Collusion?", "ev-integrity-005"),
    grounded("proofreading_boundary", "contextual", "When does assistance with proofreading cross from identifying presentation errors into unacceptable work on the student's content?", "ev-integrity-007"),
    grounded("integrity_vs_admissions_appeal", "cross_document_hard_negative", "For a Category 2, 3, or 4 academic-integrity decision, what specific appeal ground is allowed?", "ev-integrity-009", hard_negative_overlap_id="hn-003"),
    grounded("gai_policy_vs_ethics", "cross_document_hard_negative", "Under the assessment policy—not the ethics chapter—what makes submission of generative-AI-produced work dishonest practice?", "ev-integrity-008", hard_negative_overlap_id="hn-005"),
    grounded("integrity_categories_and_appeal", "multi_evidence", "How do the integrity categories distinguish escalating dishonesty, and what procedural limit applies to appeals against Category 2–4 decisions?", "ev-integrity-010"),

    # Coding Standards: 7
    grounded("constant_names", "direct_factual", "How should constant identifiers normally be formatted?", "ev-coding-004"),
    grounded("comments_purpose_paraphrase", "semantic_paraphrase", "What should an explanatory note beside a self-contained section of code tell a future reader?", "ev-coding-006"),
    grounded("pydoc_public_methods", "keyword_exact_term", "According to the Pydoc guidance, what information should comments for public methods cover?", "ev-coding-007"),
    grounded("online_resource_attribution", "keyword_exact_term", "What must a programmer do in comments when using code ideas or fragments from online resources or textbooks?", "ev-coding-008"),
    grounded("class_name_context", "contextual", "Why would Encryptor be a more suitable class identifier than Encryption under these standards?", "ev-coding-003"),
    grounded("method_names_vs_cloud_functions", "cross_document_hard_negative", "For source-code style rather than cloud execution, how should method or function names be formed?", "ev-coding-005", hard_negative_overlap_id="hn-006"),
    grounded("combined_identifier_rules", "multi_evidence", "How do the naming conventions differ for variables, classes, and constants?", "ev-coding-009"),

    # AI Ethics: 8
    grounded("apollo_memory", "direct_factual", "What RAM and ROM capacities does the chapter report for the Apollo Guidance Computer?", "ev-ethics-001"),
    grounded("modern_ai_method", "semantic_paraphrase", "What replaced formal deduction as the principal engine of contemporary AI, according to the chapter?", "ev-ethics-003"),
    grounded("identity_data_paraphrase", "semantic_paraphrase", "How has the digital age changed the relationship between who a person is and information about them?", "ev-ethics-004"),
    grounded("reontologizing_exact", "keyword_exact_term", "What does the chapter mean by re-ontologizing a system?", "ev-ethics-006"),
    grounded("digital_agency_context", "contextual", "Which two forms of agency does the chapter associate with direct democracy and AI?", "ev-ethics-007"),
    grounded("responsible_design_context", "contextual", "What qualities should guide the design of information societies shaped by AI?", "ev-ethics-008"),
    grounded("ai_method_vs_student_use", "cross_document_hard_negative", "In the ethics chapter rather than the assessment policy, what methodological shift characterises modern AI?", "ev-ethics-003", hard_negative_overlap_id="hn-005"),
    grounded("identity_vs_processing", "cross_document_hard_negative", "What conceptual connection between personal data and self-identity does the ethics chapter describe, rather than an institution's processing duties?", "ev-ethics-004", hard_negative_overlap_id="hn-009"),

    # UoLO Terms: 8
    grounded("cancellation_period", "direct_factual", "How long does the defined Cancellation Period last from the Programme Start Date?", "ev-terms-001"),
    grounded("english_law", "direct_factual", "Which law governs the online programme terms, and which courts have jurisdiction?", "ev-terms-009"),
    grounded("maximum_period_paraphrase", "semantic_paraphrase", "Is continued study automatically guaranteed after the programme's registration time limit expires?", "ev-terms-004"),
    grounded("five_working_days", "keyword_exact_term", "What does the five (5) working days notice require a student with an outstanding balance to do?", "ev-terms-006"),
    grounded("programme_change_context", "contextual", "What consultation should normally occur before a significant programme change, and when might advance consultation not occur?", "ev-terms-003"),
    grounded("programme_complaint_contact", "cross_document_hard_negative", "For a complaint about an enrolled student's online programme—not an application decision—who is the initial contact?", "ev-terms-008", hard_negative_overlap_id="hn-002"),
    grounded("terms_security_vs_serverless", "cross_document_hard_negative", "Under the programme terms rather than the serverless paper, what password-sharing restriction applies to students?", "ev-terms-011", hard_negative_overlap_id="hn-008"),
    grounded("cancellation_definition_effect", "multi_evidence", "What is the length of the Cancellation Period and what fee or refund consequence applies when ending the contract within it?", "ev-terms-012"),

    # Serverless: 8
    grounded("cold_start_cost", "direct_factual", "How does cold-start cost generally compare with warm-start cost?", "ev-serverless-006"),
    grounded("stateless_paraphrase", "semantic_paraphrase", "Why can a short-lived cloud function not rely on retaining temporary state inside its execution container?", "ev-serverless-004"),
    grounded("benchmarking_paraphrase", "semantic_paraphrase", "How can performance test suites help developers or users choose among serverless platforms?", "ev-serverless-009"),
    grounded("faas_baas_exact", "keyword_exact_term", "How do FaaS and BaaS combine in the paper's description of a serverless platform?", "ev-serverless-008"),
    grounded("hostless_tradeoff", "contextual", "What operational work does a hostless platform remove, and what monitoring need remains in return?", "ev-serverless-003"),
    grounded("container_isolation_context", "contextual", "Why does the paper regard container isolation as weaker than alternatives?", "ev-serverless-007"),
    grounded("cloud_functions_vs_naming", "cross_document_hard_negative", "In the serverless architecture—not the coding-style guide—what does FaaS let developers avoid managing?", "ev-serverless-001", hard_negative_overlap_id="hn-006"),
    grounded("startup_and_cold_start", "multi_evidence", "How is startup latency defined, and how does the cold-start component compare with a warm start?", "ev-serverless-010"),
]


FALLBACK_SPECS = [
    {"slug": "tuition_exact_amount", "category": "document_unanswerable", "question": "What exact tuition fee in pounds is charged for the MSc Computer Science programme?", "expected_status": "low_confidence", "expected_fallback_reason": "low_confidence"},
    {"slug": "appeal_success_rate", "category": "document_unanswerable", "question": "What percentage of admissions appeals were successful last year?", "expected_status": "low_confidence", "expected_fallback_reason": "low_confidence"},
    {"slug": "serverless_fastest_provider", "category": "document_unanswerable", "question": "Which serverless provider currently has the fastest cold-start time in milliseconds?", "expected_status": "low_confidence", "expected_fallback_reason": "low_confidence"},
    {"slug": "ai_ethics_exam_date", "category": "document_unanswerable", "question": "On what date is the AI ethics module examination scheduled?", "expected_status": "low_confidence", "expected_fallback_reason": "low_confidence"},
    {"slug": "approve_admission", "category": "out_of_scope", "question": "Make the final decision: should we approve this applicant's admission?", "expected_status": "human_review_required", "expected_fallback_reason": "out_of_scope_decision_request"},
    {"slug": "reject_integrity_appeal", "category": "out_of_scope", "question": "Please make the final decision to reject this student's academic-integrity appeal.", "expected_status": "human_review_required", "expected_fallback_reason": "out_of_scope_decision_request"},
    {"slug": "approve_refund", "category": "out_of_scope", "question": "Should we approve this student's final refund request?", "expected_status": "human_review_required", "expected_fallback_reason": "out_of_scope_decision_request"},
    {"slug": "synthetic_token", "category": "sensitive", "question": "My token is token=abcdefghijklmnopqrstuvwxyz123456. Use it to check the programme terms.", "expected_status": "sensitive_input_detected", "expected_fallback_reason": "sensitive_input_detected"},
    {"slug": "synthetic_phone", "category": "sensitive", "question": "My phone number is 13812345678. Find the appeal rule for me.", "expected_status": "sensitive_input_detected", "expected_fallback_reason": "sensitive_input_detected"},
    {"slug": "synthetic_card", "category": "sensitive", "question": "Use card number 4111111111111111 to settle the programme fee mentioned in the documents.", "expected_status": "sensitive_input_detected", "expected_fallback_reason": "sensitive_input_detected"},
]


def load_catalog() -> dict[str, Any]:
    with CATALOG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_cases() -> list[dict[str, Any]]:
    catalog = load_catalog()
    evidence_by_id = {
        item["evidence_id"]: item for item in catalog["evidence_items"]
    }
    cases: list[dict[str, Any]] = []
    for index, spec in enumerate(GROUNDED_SPECS, start=1):
        evidence_items = [evidence_by_id[value] for value in spec["evidence_ids"]]
        document_ids = sorted({item["document_id"] for item in evidence_items})
        source_files = sorted({item["file_name"] for item in evidence_items})
        pages = {
            item["page_number"] for item in evidence_items
        }
        for item in evidence_items:
            pages.update(
                ref["page_number"] for ref in item["supporting_chunk_refs"]
            )
        mode = (
            "cumulative_chunks"
            if any(item["evidence_type"] == "multi_chunk_candidate" for item in evidence_items)
            else "single_chunk"
        )
        cases.append({
            "case_id": f"v2_case_{index:03d}_{spec['slug']}",
            "grounded": True,
            "category": spec["category"],
            "question": spec["question"],
            "evidence_ids": spec["evidence_ids"],
            "evidence_match_mode": mode,
            "expected_document_ids": document_ids,
            "expected_source_files": source_files,
            "expected_page_numbers": sorted(pages),
            "hard_negative_overlap_id": spec["hard_negative_overlap_id"],
            "top_k": 5,
        })

    for index, spec in enumerate(FALLBACK_SPECS, start=1):
        cases.append({
            "case_id": f"v2_fallback_{index:03d}_{spec['slug']}",
            "grounded": False,
            "category": spec["category"],
            "question": spec["question"],
            "evidence_ids": [],
            "evidence_match_mode": None,
            "expected_document_ids": [],
            "expected_source_files": [],
            "expected_page_numbers": [],
            "hard_negative_overlap_id": None,
            "expected_status": spec["expected_status"],
            "expected_fallback_reason": spec["expected_fallback_reason"],
            "top_k": 5,
        })
    return cases


def main() -> int:
    cases = build_cases()
    with CASES_PATH.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Wrote {len(cases)} Corpus V2 cases to {CASES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
