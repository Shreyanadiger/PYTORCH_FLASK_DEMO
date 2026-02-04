"""
Text Generator Runner
Loads the text_generator.pth model and generates text.
"""

import torch
import torch.nn as nn
import os

# Define the TextGenerator model architecture
# Based on model dimensions: vocab_size=5273, embed_size=16, hidden_size=32, num_layers=1
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size=16, hidden_size=32, num_layers=1):
        super(TextGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        out, hidden = self.lstm(embed, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)


# Global variables
model = None
word_to_idx = None
idx_to_word = None
device = None
VOCAB_SIZE = 5273


def create_simple_vocab():
    """Create a simple vocabulary based on common words."""
    # Common English words - this is a placeholder
    # In practice, you would need the original vocabulary used during training
    words = ['<unk>', '<pad>', '<eos>', '<sos>']
    
    # Add common words (this is just an approximation)
    common_words = [
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'to',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours',
        'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose',
        'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then',
        'and', 'but', 'or', 'if', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
        'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'from', 'up', 'down', 'in', 'out', 'on',
        'off', 'over', 'under', 'again', 'further', 'once', 'say', 'said', 'see', 'go',
        'come', 'make', 'know', 'get', 'give', 'take', 'think', 'tell', 'find', 'want',
        'look', 'use', 'say', 'good', 'new', 'first', 'last', 'long', 'great', 'little',
        'own', 'old', 'right', 'big', 'high', 'different', 'small', 'large', 'next',
        'early', 'young', 'important', 'few', 'public', 'bad', 'same', 'able',
        'man', 'woman', 'child', 'world', 'life', 'hand', 'part', 'place', 'case',
        'week', 'company', 'system', 'program', 'question', 'work', 'government',
        'number', 'night', 'point', 'home', 'water', 'room', 'mother', 'area',
        'money', 'story', 'fact', 'month', 'lot', 'right', 'study', 'book', 'eye',
        'job', 'word', 'business', 'issue', 'side', 'kind', 'head', 'house', 'service',
        'friend', 'father', 'power', 'hour', 'game', 'line', 'end', 'member', 'law',
        'car', 'city', 'community', 'name', 'president', 'team', 'minute', 'idea',
        'kid', 'body', 'information', 'back', 'parent', 'face', 'others', 'level',
        'office', 'door', 'health', 'person', 'art', 'war', 'history', 'party',
        'result', 'change', 'morning', 'reason', 'research', 'girl', 'guy', 'moment',
        'air', 'teacher', 'force', 'education', 'love', 'time', 'day', 'year', 'way',
        'thing', 'people', 'state', 'country', 'school', 'family', 'student', 'group',
        'problem', 'today', 'tomorrow', 'yesterday'
    ]
    words.extend(common_words)
    
    # Fill remaining vocab with numbered tokens
    while len(words) < VOCAB_SIZE:
        words.append(f'<word_{len(words)}>')
    
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    idx_to_word = {idx: word for idx, word in enumerate(words)}
    
    return word_to_idx, idx_to_word


def load_model():
    """Load the trained model from checkpoint."""
    global model, word_to_idx, idx_to_word, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Path to the model file
    model_path = os.path.join(os.path.dirname(__file__), "text generator.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    # Create vocabulary
    word_to_idx, idx_to_word = create_simple_vocab()
    
    # Create model with correct architecture
    model = TextGenerator(VOCAB_SIZE, embed_size=16, hidden_size=32, num_layers=1)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully! Vocabulary size: {VOCAB_SIZE}")
    return True


def generate_text(model, seed_text, num_words=100, temperature=0.8):
    """
    Generate text using the trained model.
    
    Args:
        model: The trained TextGenerator model
        seed_text: Initial text to start generation
        num_words: Number of words to generate
        temperature: Controls randomness (lower = more predictable)
    
    Returns:
        Generated text string
    """
    global word_to_idx, idx_to_word, device
    
    model.eval()
    
    # Tokenize seed text
    words = seed_text.lower().split()
    indices = [word_to_idx.get(w, word_to_idx.get('<unk>', 0)) for w in words]
    
    if not indices:
        indices = [0]  # Use <unk> if no valid words
    
    hidden = model.init_hidden(1, device)
    
    # Process seed text
    for idx in indices[:-1]:
        x = torch.tensor([[idx]]).to(device)
        _, hidden = model(x, hidden)
    
    # Start generation from the last word of seed
    x = torch.tensor([[indices[-1]]]).to(device)
    generated_words = words.copy()
    
    with torch.no_grad():
        for _ in range(num_words):
            output, hidden = model(x, hidden)
            
            # Apply temperature scaling
            output = output[:, -1, :] / temperature
            probs = torch.softmax(output, dim=-1)
            
            # Sample from the probability distribution
            idx = torch.multinomial(probs, 1).item()
            
            # Convert index to word
            word = idx_to_word.get(idx, '<unk>')
            generated_words.append(word)
            
            # Prepare next input
            x = torch.tensor([[idx]]).to(device)
    
    return ' '.join(generated_words)


# Load model and run when script is executed
if __name__ == "__main__":
    if load_model():
        print('Generated text is: ', generate_text(model, 'can I', num_words=100))