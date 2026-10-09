# Data card: Puente benchmark (`data/cases.jsonl`)

**What it is.** 296 short Spanish patient messages, each with the structured English intake it should produce.
It is a synthetic, template-built test suite for one question: does a system that turns Spanish patient messages
into an English intake get clinical-Spanish pitfalls right, without breaking the ordinary cases?

**What it is not.** It is not real patient data and not a clinical validation. A good score here is necessary,
not sufficient, for safe use.

## How it is built

- `bench/build.py` writes every case from templates with a fixed seed; `python -m bench.build` regenerates the file
  exactly (CI checks this). Gold labels come from the templates only; no model wrote or labelled any case.
- Each pitfall comes as a minimal set: the misleading word in its real sense, a **control** where the literal
  reading is right, and, for *intoxicado*, **ambiguous** cases where the correct output is to ask the clinician to
  confirm. A system that always 'corrects' the word fails the controls.
- Ordinary sentences are mixed in at random: denied symptoms ("No tengo fiebre"), medications with dose and
  frequency, allergies, conditions, pregnancy status, emergency signs and Spanglish ("Me salió un rash").
- `checks` in each case list what must be present and what must be absent for the pitfall to count as passed;
  `bench/score.py` applies them deterministically.

## Families and sources

| Family | In the pitfall guard? | Cases (variant: count) | Meaning used | Sources |
|---|---|---|---|---|
| intoxicado | yes | alcohol: 12, ambiguous: 8, drug: 8, food: 18 | Spanish 'intoxicado/a' means poisoned or made sick by something taken in (food, a chemical, a medicine, alcohol). It does not by itself mean drunk or high. Decide from context: after food -> food_poisoning; after alcohol -> alcohol_intoxication; after pills or drugs -> drug_overdose. If the context does not say what was taken, list 'intoxicado' under needs_clarification. | OpenExamPrep, CCHI Certified Healthcare Interpreter study guide, 'Abbreviations, Eponyms, False Cognates & Terminology Management' (open-exam-prep.com/study-guides/cchi-chi/terminology-word-structure/abbreviations-eponyms-false-cognates-terminology-management); J. DeCamp, 'When Intoxicado ≠ Intoxicated: Avoiding False Friends and Critical Consequences in Machine Translation', MITRE, 2017 (kde.mitre.org/?p=3911); J Gen Intern Med 2024, doi:10.1007/s11606-024-08619-8 (case in which 'intoxicado' led to an overdose work-up) |
| embarazada | yes | control: 12, pitfall: 12 | Spanish 'embarazada' means pregnant, not embarrassed (embarrassed = 'avergonzada', 'me da vergüenza'). | OpenExamPrep, CCHI Certified Healthcare Interpreter study guide, 'Abbreviations, Eponyms, False Cognates & Terminology Management' (open-exam-prep.com/study-guides/cchi-chi/terminology-word-structure/abbreviations-eponyms-false-cognates-terminology-management); N. Tavarez, 'The Danger of False Cognates in Healthcare', ALTA Language Services (altalang.com/beyond-words/false-cognates-healthcare/) |
| azucar | yes | control: 6, diabetes: 12, low: 6 | Colloquially 'tengo azúcar' / 'padezco del azúcar' means the person has diabetes (high blood sugar). 'Se me bajó el azúcar' means low blood sugar. Sugar in food or drink is just sugar. | Common Ground International, 'How to Discuss Signs and Symptoms of Diabetes in Spanish' (commongroundinternational.com/medical-spanish-blog/discuss-signs-symptoms-diabetes-spanish/); Inklingo, 'How to say I am diabetic in Spanish' (inklingo.app) |
| nervios | yes | control: 6, pitfall: 10 | 'Ataque de nervios' is a recognised Latino cultural idiom of distress. Keep the Spanish term under culture_bound for the clinician; do not translate it into a diagnosis. | López-Cepero et al., Soc Psychiatry Psychiatr Epidemiol 2023, doi:10.1007/s00127-023-02601-1; Lerman Ginzburg et al., BMC Psychol 2021, doi:10.1186/s40359-021-00544-3; Park & Kim, Adv Exp Med Biol 2020, doi:10.1007/978-981-32-9705-0_12 (DSM-5 cultural concepts of distress) |
| constipado | **no (held out)** | control: 12, pitfall: 12 | Spanish 'constipado/a' means having a cold (nasal congestion), not constipated (= 'estreñido/a'). | OpenExamPrep, CCHI Certified Healthcare Interpreter study guide, 'Abbreviations, Eponyms, False Cognates & Terminology Management' (open-exam-prep.com/study-guides/cchi-chi/terminology-word-structure/abbreviations-eponyms-false-cognates-terminology-management); N. Tavarez, 'The Danger of False Cognates in Healthcare', ALTA Language Services (altalang.com/beyond-words/false-cognates-healthcare/) |
| boca_estomago | **no (held out)** | control: 6, pitfall: 12 | 'La boca del estómago' is the pit of the stomach (epigastrium), not the mouth. | Real Academia Española, Diccionario de la lengua española, entry 'estómago' (dle.rae.es/estómago): set phrase 'boca del estómago' |

Plus 144 general intake cases with no pitfall word.

The two held-out families are never shown to the guard, so the benchmark can tell whether the guard helps only on
the words it was given.

Sources disagree on details of the widely cited 'intoxicado' malpractice case (paraplegic vs quadriplegic, the
settlement amount), so this project cites it only as an example of the word being misread.

## Limitations

- The Spanish templates were written by the author with AI assistance and have **not yet been reviewed by a native
  Spanish-speaking clinician or interpreter**. Review of the pitfall items by one is the most valuable next step.
- Short, clean, written messages. Real speech adds recognition errors, and real patients use far more regional
  vocabulary than six word families.
- Regional usage varies; the meanings above follow the cited sources, which mostly describe Mexican and general
  Latin American Spanish as met in US healthcare.
