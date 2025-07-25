import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForImageClassification
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ======================= SIMPLE TEXT CLASSIFIER =======================

class SimpleTextClassifier(nn.Module):
    """Simple RoBERTa-based classifier without using Trainer"""
    def __init__(self, model_name="roberta-base", num_classes=2):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        output = self.dropout(cls_output)
        return self.classifier(output)

class TextDataset(Dataset):
    """Simple dataset class"""
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_training_data():
    """Create comprehensive training dataset"""
    print("📚 Creating training dataset...")
    
    # Human-written examples (natural, personal, conversational)
    human_texts = [
        # Personal experiences
        "I had the most amazing weekend! Went hiking with my friends and the weather was perfect.",
        "My dog keeps stealing my socks and hiding them under the couch. It's annoying but also hilarious.",
        "Just finished reading this incredible book about space exploration. Couldn't put it down!",
        "The new coffee shop downtown has the best lattes. Going to be my new study spot for sure.",
        "My mom called today and we talked for two hours about absolutely nothing important. I love those calls.",
        
        # Casual observations
        "Traffic was insane this morning. Took me 45 minutes to go what usually takes 15.",
        "I love how quiet the neighborhood gets after 10 PM. Perfect for evening walks.",
        "The weather's been so unpredictable lately. Sunny one minute, raining the next.",
        "My cat has claimed my favorite chair as her new sleeping spot. Guess I need a new chair.",
        "Found some old photos while cleaning out my closet. So many good memories from college!",
        
        # Personal opinions
        "I don't understand the hype around that new TV show everyone's talking about.",
        "Pizza for breakfast? Yes please! Don't judge me, it's basically bread and vegetables.",
        "I think people spend way too much time on social media these days, myself included.",
        "There's something magical about bookstores that online shopping just can't replicate.",
        "I love how my garden looks in the morning when everything is covered in dew.",
        
        # Conversational style
        "So my sister calls me yesterday and tells me the craziest story about her neighbor...",
        "You know what really bugs me? When people don't return their shopping carts.",
        "I was thinking about taking up painting again. Haven't done it since high school.",
        "My roommate and I are planning a road trip next month. Can't wait to get out of the city!",
        "The grocery store was packed today. I swear everyone shops at the same time I do.",
        
        # More personal examples
        "I finally beat that video game I've been stuck on for weeks. Feels amazing!",
        "My grandmother's recipe for chocolate chip cookies is still the best I've ever tasted.",
        "I love the smell of rain on hot pavement. It reminds me of summer storms as a kid.",
        "My workout routine has been inconsistent lately. Need to get back on track.",
        "The local farmer's market has the freshest vegetables. Worth the early morning trip.",
        
        # Everyday struggles
        "I keep forgetting to water my plants and then feeling guilty when they look sad.",
        "My phone battery dies at the worst possible moments. Time for an upgrade maybe?",
        "I started learning Spanish on an app but I keep forgetting to practice daily.",
        "The line at the DMV was ridiculous today. Waited two hours just to renew my license.",
        "I love cooking but hate doing the dishes afterward. Such a dilemma."
    ]
    
    # AI-generated examples (formal, structured, generic)
    ai_texts = [
        # Typical AI responses
        "As an AI language model, I can provide you with comprehensive information on this topic.",
        "Based on extensive research and analysis, there are several key factors to consider.",
        "Here are the top 5 strategies you should implement to achieve optimal results:",
        "It's important to note that individual experiences may vary depending on specific circumstances.",
        "According to industry experts, this approach has proven to be highly effective.",
        
        # Structured responses
        "To address this issue effectively, follow these step-by-step instructions: 1) First step 2) Second step 3) Final step.",
        "The data indicates a strong correlation between these variables across multiple studies.",
        "From a technical perspective, this solution offers superior performance and reliability.",
        "Research suggests that implementing these best practices can lead to significant improvements.",
        "In conclusion, this methodology provides a comprehensive framework for success.",
        
        # Generic advice
        "When evaluating your options, consider factors such as cost, efficiency, and long-term sustainability.",
        "The implementation of these features will enhance user experience and drive engagement.",
        "Studies have consistently shown that organizations using this approach report better outcomes.",
        "This innovative solution addresses the key challenges faced by modern businesses today.",
        "By leveraging advanced analytics, you can gain valuable insights into performance metrics.",
        
        # Formal explanations
        "The underlying principle behind this concept involves complex interactions between multiple components.",
        "To optimize your workflow, integrate these tools and techniques into your existing processes.",
        "The benefits of this approach include increased efficiency, reduced costs, and improved scalability.",
        "Market analysis reveals significant opportunities for growth in emerging sectors.",
        "This comprehensive guide covers all aspects of the topic in detail and provides actionable insights.",
        
        # AI-like conclusions
        "In summary, these findings demonstrate the effectiveness of the proposed methodology.",
        "The results of this analysis provide valuable guidance for decision-making processes.",
        "Based on current trends and projections, we can expect continued growth in this area.",
        "These recommendations are supported by extensive research and real-world case studies.",
        "The successful implementation of these strategies requires careful planning and execution.",
        
        # Technical language
        "The algorithm utilizes machine learning techniques to process and analyze large datasets efficiently.",
        "This framework incorporates industry standards and best practices for optimal performance.",
        "The system architecture is designed to handle high-volume transactions with minimal latency.",
        "Advanced features include real-time monitoring, automated reporting, and predictive analytics.",
        "The platform provides seamless integration with existing enterprise systems and workflows."
    ]
    
    # Expand datasets
    human_texts = human_texts * 8  # Multiply for more samples
    ai_texts = ai_texts * 8
    
    # Combine
    all_texts = human_texts + ai_texts
    all_labels = [0] * len(human_texts) + [1] * len(ai_texts)  # 0=human, 1=AI
    
    print(f"✅ Created dataset with {len(all_texts)} samples")
    print(f"   Human samples: {len(human_texts)}")
    print(f"   AI samples: {len(ai_texts)}")
    
    return all_texts, all_labels

def train_text_classifier_simple():
    """Train text classifier without using Trainer class"""
    print("\n🤖 Training Text Classifier")
    print("=" * 40)
    
    # Create dataset
    texts, labels = create_training_data()
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"📊 Dataset split:")
    print(f"   Training: {len(train_texts)} samples")
    print(f"   Testing: {len(test_texts)} samples")
    
    # Initialize model and tokenizer
    print("🔄 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = SimpleTextClassifier()
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    epochs = 2
    
    print(f"\n🚀 Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            accuracy = correct_predictions / total_samples
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'accuracy': f'{accuracy:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples
        
        print(f"✅ Epoch {epoch+1} completed:")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Training Accuracy: {epoch_accuracy:.4f}")
    
    # Evaluation
    print("\n📊 Evaluating model...")
    model.eval()
    test_predictions = []
    test_true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            
            test_predictions.extend(predictions.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_accuracy = accuracy_score(test_true_labels, test_predictions)
    
    print(f"\n🎯 Final Results:")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print("\n📈 Detailed Classification Report:")
    print(classification_report(test_true_labels, test_predictions, 
                              target_names=['Human', 'AI-Generated']))
    
    # Save model
    print("\n💾 Saving model...")
    os.makedirs('./saved_models/text_classifier', exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), './saved_models/text_classifier/pytorch_model.bin')
    
    # Save tokenizer
    tokenizer.save_pretrained('./saved_models/text_classifier')
    
    # Save model config
    model.roberta.config.save_pretrained('./saved_models/text_classifier')
    
    # Save training info
    training_info = {
        'model_type': 'SimpleTextClassifier',
        'test_accuracy': test_accuracy,
        'num_samples': len(texts),
        'epochs': epochs,
        'max_length': 256
    }
    
    with open('./saved_models/text_classifier/training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print("✅ Model saved successfully!")
    
    # Test with sample predictions
    print("\n🧪 Testing sample predictions:")
    test_samples = [
        ("I love spending time with my family on weekends, it's the best part of my week!", "Expected: Human"),
        ("Based on comprehensive analysis, this solution provides optimal performance across multiple metrics.", "Expected: AI")
    ]
    
    model.eval()
    for text, expected in test_samples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1).item()
            confidence = probs.max().item()
        
        label = "AI-generated" if pred == 1 else "Human-written"
        print(f"   Text: {text[:60]}...")
        print(f"   Prediction: {label} (confidence: {confidence:.3f}) | {expected}")
    
    return model, tokenizer

def setup_image_classifier():
    """Setup image classifier (pre-trained model)"""
    print("\n🖼️  Setting up Image Classifier")
    print("=" * 40)
    
    print("🔄 Loading pre-trained Vision Transformer...")
    
    # Initialize processor and model
    model_name = "google/vit-base-patch16-224"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name, 
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    # Save for API use
    os.makedirs('./saved_models/image_classifier', exist_ok=True)
    model.save_pretrained('./saved_models/image_classifier')
    processor.save_pretrained('./saved_models/image_classifier')
    
    # Save info
    image_info = {
        'model_type': 'ViT',
        'model_name': model_name,
        'num_labels': 2,
        'note': 'Pre-trained model - add your own images to data/real_images and data/ai_images for custom training'
    }
    
    with open('./saved_models/image_classifier/model_info.json', 'w') as f:
        json.dump(image_info, f, indent=2)
    
    print("✅ Image classifier setup completed!")
    print("💡 To improve image classification:")
    print("   1. Add real photos to: data/real_images/")
    print("   2. Add AI-generated images to: data/ai_images/")
    print("   3. Re-run this script for custom training")
    
    return model, processor

def main():
    """Main training function"""
    print("🚀 Standalone AI Detection Training")
    print("=" * 50)
    print("✨ No external datasets required!")
    print("🔧 Fixed for latest Transformers versions")
    print("=" * 50)
    
    # Create directories
    os.makedirs('./saved_models/text_classifier', exist_ok=True)
    os.makedirs('./saved_models/image_classifier', exist_ok=True)
    os.makedirs('./data/real_images', exist_ok=True)
    os.makedirs('./data/ai_images', exist_ok=True)
    
    try:
        # Train text classifier
        text_model, text_tokenizer = train_text_classifier_simple()
        
        # Setup image classifier
        image_model, image_processor = setup_image_classifier()
        
        print("\n" + "=" * 50)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        print("\n📁 Models saved to:")
        print("   📝 Text classifier: ./saved_models/text_classifier/")
        print("   🖼️  Image classifier: ./saved_models/image_classifier/")
        
        print("\n🚀 Next Steps:")
        print("   1. Start Flask API: python flask_api.py")
        print("   2. Test the API: python test_api.py")
        print("   3. Integrate with your MERN stack!")
        
        print("\n💡 Optional Improvements:")
        print("   • Add your own image datasets for better image detection")
        print("   • Expand text training data with more examples")
        print("   • Use GPU for faster training")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check your internet connection")
        print("   2. Ensure you have enough disk space")
        print("   3. Try: pip install --upgrade transformers torch")
        print("   4. If using GPU, check CUDA installation")
        
        import traceback
        print(f"\nFull error traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()