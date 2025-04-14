from agents.paper_finder import PaperFinder
from agents.key_extractor import KeyExtractor
from agents.answer_synthesizer import AnswerSynthesizer
import argparse

def main():
    parser = argparse.ArgumentParser(description="Medical Research Agent")
    parser.add_argument("--query", required=True, help="Medical research question")
    args = parser.parse_args()

    # Initialize agents
    finder = PaperFinder()
    extractor = KeyExtractor()
    synthesizer = AnswerSynthesizer()

    # Pipeline execution
    papers = finder.find_relevant_papers(args.query)
    key_passages = extractor.extract_passages(papers, args.query)
    final_answer = synthesizer.synthesize_answer(key_passages, args.query)

    print(f"\nFinal Answer:\n{final_answer}")

if __name__ == "__main__":
    main()
