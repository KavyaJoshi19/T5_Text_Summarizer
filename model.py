# model.py
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5SummarizationModel:
    def __init__(self, model_name="t5-base", device=None):
        """
        Initialize the T5 summarization model.
        
        Args:
            model_name (str): The name or path of the T5 model to use.
                Options: t5-small, t5-base, t5-large, t5-3b, t5-11b
            device (str, optional): Device to use for computation ('cuda' or 'cpu').
                If None, will use CUDA if available.
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading T5 model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Move model to device
        self.model.to(self.device)
    
    def summarize(self, text, max_length=150, min_length=40, 
                  num_beams=4, early_stopping=True, no_repeat_ngram_size=2):
        """
        Generate a summary for the input text.
        
        Args:
            text (str): The text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            num_beams (int): Number of beams for beam search
            early_stopping (bool): Whether to stop beam search when at least num_beams 
                                  sentences are finished per batch
            no_repeat_ngram_size (int): Size of n-grams that cannot be repeated in the generation
        
        Returns:
            str: The generated summary
        """
        # Prepare input for T5
        input_text = "summarize: " + text.strip()
        
        # Tokenize the input
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", 
                                      max_length=1024, truncation=True)
        inputs = inputs.to(self.device)
        
        # Generate summary
        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            no_repeat_ngram_size=no_repeat_ngram_size
        )
        
        # Decode the summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary
    
    def save_model(self, path):
        """Save the model and tokenizer to the specified path."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load the model and tokenizer from the specified path."""
        self.tokenizer = T5Tokenizer.from_pretrained(path)
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.model.to(self.device)
        print(f"Model loaded from {path}")

# Example usage
if __name__ == "__main__":
    # Initialize the model
    summarizer = T5SummarizationModel("t5-small")  # Use t5-small for testing as it's faster
    
    # Test text for summarization
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to 
    the natural intelligence displayed by animals including humans. AI research has been defined 
    as the field of study of intelligent agents, which refers to any system that perceives its 
    environment and takes actions that maximize its chance of achieving its goals. The term 
    "artificial intelligence" had previously been used to describe machines that mimic and display 
    "human" cognitive skills that are associated with the human mind, such as "learning" and 
    "problem-solving". This definition has since been rejected by major AI researchers who now 
    describe AI in terms of rationality and acting rationally, which does not limit how 
    intelligence can be articulated.
    """
    
    # Generate summary
    summary = summarizer.summarize(text)
    
    print("\nInput text:")
    print(text)
    print("\nGenerated summary:")
    print(summary)