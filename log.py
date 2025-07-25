import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler,Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
import random
import json
import logging
try:
    logging.basicConfig(filename="sspam_ham_trainning.log",level=logging.INFO,format='%(asctime)s-(levelname)s-%(message)s')
    logging.info("logging configured sucessfully.")
except Exception as e:
    print(f"CRITICAL ERROR:Failed to configure loggging : {e}")
    exit()
url=r"C:\Users\user\Documents\neural_network\spamham.csv"
link=r"C:\Users\user\Documents\neural_network\email.csv"
try:
    logging.info(f" Attempting to load data from {url} and {link}")
    df=pd.read_csv(url)
    extr_df=pd.read_csv(link)
    logging.info("Data loaded succesfully.")
    #normalize column names for extra_df
    extra_df=extr_df.rename(columns={'Category':'Label','Message':'Text'})
    logging.info("Renamed columns in extra_df.")
    #map labels to numerical values
    df['Label']=df['Label'].map({'ham':0,'spam':1})
    extra_df['Label']=extra_df['Label'].map({'ham':0,'spam':1})
    logging.info("Mapped labels to numerical values")
    extra_spam=extra_df[extra_df['Label']==1]
    logging.info(f"Extracted {len(extra_spam)}")
    #concatenate original df with extra spam messages
    df=pd.concat([df,extra_spam],ignore_index=True)
    logging.info(f"Combined datasets. Total messages:{len(df)}")
    #shuffle the combined dataframe
    df=df.sample(frac=1.0,random_state=42).reset_index(drop=True)
    logging.info("Dataframe shuffled successfully")
except FileNotFoundError as e:
    logging.critical(f"CRITICAL ERROR:One or both data files not found. Check paths:{e}")
    exit()
except KeyError as e:
    logging.error(f"ERROR:Missing expected column during data loading/renaming:{e}")
    exit()
try:
    new_spam_messages=[  
     "you've won a brand new iphone!Visit bit.ly/freephone to claim now",
     "Congratulations you've been selected for a $1,000 walmart gift card.Visit now to redeem.",
     "Earn $500 daily working from home! Visit workfastmoney.com to get started",
     "Your number was randomly chosen!Visit luckywinner.net to see your price!",
     "Click here to remove all your debt instantly.Visit debtfree2025.org for details",
     "Want to lose 10kg in 10 days? Visit slimquickresults.xyz for a free trail ",
     "Urgent:Your account will be suspended.Visit secure-login-alert.net to verify info now",
     "Congratulations!You've been selected for a loan of $50,000 at 0% interest. click here to claim:loannow.xyz",
     "for account related queries visit our website or speak with a support agent",
     "dear customer please visit your nearest atm to update your kyc before 30th june",
     "you can view your transaction history when you visit your oline banking portal",
     "your insurance documents are ready for pickup. kindly visit the branch during working hours",
     "congratulations you've been selected for a loan of $50,000 at 0% interest. click here to claim:loannow.xyz",
     "your bank account has been suspended verify your details immediately at secure-login.bankverify.cc",
     "final warning pay your unpaid loan today and get 50% discount. click to avoid penalty",
     "congrats! you have won #1,000 in glo promo,call now!",
     "hi dear long time! let's connect on whatsapp +234xxxxxxxx",
     "hello friend,i have business for you! big money involved",
     "need quick loan? no bvn needed instant approval",
     "please donate to orphanage,account number 2663995050 access bank",
     "new year giveaway,just reply yes to win",
     "MTN:your sim has been upgraded click to activate",
     "hello,are you interested in forex trading? dm me now!",
     "this is from zenith bank reactivate your token now.",
     "your gtb account is restricted visit this link to reactivate",
     "click now to receive #2000 MTN data bonus",
     "i'm a soldier overseas. looking for honest partner",
     "we offer zero interest loan reply yes now",
     "you have been selected as lucky winner tap here",
     "sign up and get #10,000 daily doing surveys",
     "buy laptop at giveaway price limited offer!",
     "urgent message from cac your business is flagged",
     "i'll pay you 5k just to chat,interesred?",
     "visit this link to win your iphone 16 pro max",
     "visit this site to win 100k",
     "congratulations!you are the winner of a free gift.claim your prize now.",
     "urgent:your account has been selected for a bonus offer.click the link to verify",
     "get a loan with zer0 interest today!limited time promotion,call now",
     "buy cheap forex trading packages and earn huge income",
     "you have been preapproved for a secured credit card.activate your account today.",
     "reply yes to suscribe to our exclusive giveaway",
     "discount on all products!visit our website and save big",
     "investment opportunity with guaranteed returns. don't miss out",
     "your atm card will be deactivated soon. reactivate now!",
     "earn $5,000/month with this secret crypto method no skills required",
     "final notice!your account will be suspended.verify now",
     "act now and get a 90% discount on all electronics!",
     "unlock $1,000 instantly with this exclusive app-download now!",
     "we tried to deliver your package,but need more details.click to update",
     "limited time crypto investment with 200% guaranteed returns",
     "instant approval!apply now for $5,000 credit line",
     "don't miss your last chance to win a luxury vacation. enter here",
     "update required.your system is out of date.tap to install",
     "i'm opeyemi",
     "i'm ai architect"
    ]
    new_spam_df=pd.DataFrame({
        'Text':new_spam_messages,
        'Label':[1]*len(new_spam_messages)
    })
    df=pd.concat([df,new_spam_df],ignore_index=True)
    logging.info(f"Added {len(new_spam_messages)} new spam messages. Total messages:{len(df)}")
except Exception as e:
    logging.error(f"ERROR:Failed to add new spam messages:{e}")
#convert dataframe columns to list
Texts=df['Text'].to_list()
Labels=df['Label'].to_list()
logging.info('Texts and Labels extracted from DataFrame')
#_ _ _ Text preprocessing Functions_ _ _
def clean_text(Text):
    try:
        url_pattern=r'http[s]?://\s+|www\.s+|\b\w+\.(com|net|org|xyz|cc|biz|info|io|ly)\b'
        Text=re.sub(url_pattern,'<URL',Text,flags=re.IGNORECASE)
        logging.debug(f"Cleaned text:{Text[:50]}...")
        return Text.lower()
    except Exception as e:
        logging.error(f"ERROR:Failed to clean text '{Text[:30]}...':{e}")
        return Text.lower()
def tokenize(Text):
    try:
      Text=clean_text(Text)
      tokens= re.findall(r'\b\w+\b',Text.lower())
      logging.debug(f"Tokenized text:{tokens[:10]}....")
      return tokens
    except Exception as e:
        logging.error(f"ERROR:Failed to tokenize text '{Text[:30]}...':{e}")
        return []
#_ _ _ vocabulary development _ _ _
try:
    logging.info("Building Vocabulary...")    
    vocab={"<PAD>":0,"<UNK>":1}
    idx=2         
    for   tokens in [tokenize(Text) for Text in Texts]:    
          for word in tokens:
              if word not in vocab:
                 vocab[word]=idx
                 idx+=1 
    logging.info(f"Vocabulary built. Size: {len(vocab)}" )
except Exception as e:
    logging.error(f"ERROR:Failed to build vocabulary:{e}")
    vocab={"<PAD>":0,"<UNK>":1}
    idx=2
def augment_tokens(tokens,deletion_prob=0.1):
    try:
        if len(tokens) <=1:
            return tokens
        augmented=[word for word in tokens if random.random() > deletion_prob]
        logging.debug(f"Augmented tokens(original len {len(tokens)},new len {len(augmented)}) .")
        return augmented
    except Exception as e:
        logging.error(f"Error: Failed to augment tokens: {e}")
        return tokens
def encode_tokens(tokens,vocab,max_len=50):
    try:
        idx=[vocab.get(word,vocab["<UNK>"]) for word in tokens]
        padded=idx + [0]*(max_len -len(idx))
        encoded=padded[:max_len]
        logging.debug(f"Encoded tokens (original len {len(tokens)},padded len {len(encoded)}).")
        return encoded
    except Exception as e:
        logging.error(f"ERROR:Failed to encode tokens:{e}")
        return[vocab["<PAD>"]]*max_len
class SpamHamDataset(Dataset):
    def __init__(self,Texts,Labels,vocab,max_len=50,augment=False):
        try:
            self.texts=Texts
            self.label=Labels
            self.vocab=vocab
            self.max_len=max_len
            self.augment=augment
            logging.info("SpamHamDataset initialized.")
        except Exception as e:
            logging.error(f"ERROR:Failed to initialize SpamHamDataset:{e}.")
    def __len__(self):
        try:
            return len(self.texts)
        except Exception as e:
            logging.erro(f"ERROR:Failed to get length of dataset :{e}")
            return 0
    def __getitem__(self, index):
        try:
            text=self.texts[idx]
            label=self.label[idx]
            tokens=tokenize(text)
            if self.augment:
                tokens=augment_tokens(tokens)
            encoded=encode_tokens(tokens,self.vocab,self.max_len)
            return torch.tensor(encoded,dtype=torch.long),torch.tensor(label,dtype=torch.float32)
        except IndexError:
            logging.error(f"ERROR:INDEXERROR in __getitem__ for index {idx}. Data may be inconsistent")
            return torch.zeros(self.max_len,dtype=torch.long),torch.tensor(0.0,dtype=torch.float32) 
try:
    logging.info("splitting data into trainning ")
    X_train_texts,X_val_texts,y_train,y_val=train_test_split(Texts,Labels,test_size=0.2,random_state=42)
    logging.info(f'Train/Val complete. Train Samples:{len(X_train_texts)},Val Samples:{len(X_val_texts)}')
    #tokenize and augment training data
    tokenized_X_train=[tokenize(text) for text in X_train_texts]
    augmented_tokenized_X_train=[augment_tokens(tokens) for tokens in tokenized_X_train]
    logging.info("Training texts tokenized and augmented")
    #encode augmented training data
    X_train=torch.tensor([encode_tokens(tokens,vocab,max_len=50)for tokens in augmented_tokenized_X_train],dtype=torch.long)
    y_train=torch.tensor(y_train,dtype=torch.float32)
    logging.info("Training data encoded and converted to pytorch tensors.")
    #tokenize and encode validation data
    tokenized_X_val=[tokenize(text) for text in X_val_texts]
    X_val=torch.tensor([encode_tokens(tokens,vocab,max_len=50)for tokens in tokenized_X_val],dtype=torch.long)
    y_val=torch.tensor(y_val,dtype=torch.float32)
    logging.info("Validation data encoded and converted to pytorch tensors")
except Exception as e:
    logging.error(f"ERROR:Failed during data splitting")
    exit()
    #calculating class weights
try:
    logging.info("Calculating class weights")
    Labels_unique,counts =torch.unique(y_train,return_counts=True)
    class_weights={}
    for label,count in zip(Labels_unique,counts):
        count_val=count.item()
        if count_val==0:
            logging.warning(f"WARNING:Skipping label {label.item()}")
            continue
        class_weights[int(label.item())]=1.0/count_val
    logging.info(f"Calculated class weights:{class_weights}")
    pos_weight=torch.tensor([class_weights.get(1,1.0)/class_weights.get(0,1.0)])
    logging.info(f"Positive weight for BCEwithLogitsLoss:{pos_weight.item():.4f}")
    #assign sample weights based on model
    samples_weight=torch.tensor([class_weights.get(int(label),1.0)for label in y_train])
    logging.info("Sample weights assigned for trainning data.")
    #create a weightedrandomsampler
    sampler=WeightedRandomSampler(weights=samples_weight,num_samples=len(samples_weight))
    logging.info("Weighted Random Sampler created.")
    #craete dataloaders for batching
    batch_size=32
    train_loader=DataLoader(TensorDataset(X_train,y_train),batch_size=batch_size,sampler=sampler)
    val_loader=DataLoader(TensorDataset(X_val,y_val),batch_size=batch_size,shuffle=False)
    logging.info(f"DataLoader created with batch size::{batch_size}")
except Exception as e:
    logging.error(f"ERROR:Failed during class weights calculation or DataLoader creation:{e}")
    exit()
class SpamHamModel(nn.Module):
    def __init__(self,vocab_size,embedded_dim,hidden_dim,output_dim,dropout_prob=0.3):
        super(SpamHamModel,self).__init__()
        try:     
          self.embedding=nn.Embedding(vocab_size,embedded_dim)
          self.lstm=nn.LSTM(embedded_dim,hidden_dim,batch_first=True,bidirectional=True)
          self.dropout=nn.Dropout(dropout_prob)
          self.fc=nn.Linear(hidden_dim*2,output_dim)
          logging.info("SpamHamModel layers initialized")
        except Exception as e:
            logging.error(f"ERROR:Failed to initialize SpamHamModel layers:{e}")
    def forward(self,x):
        try:
            embedded=self.embedding(x)
            output,_=self.lstm(embedded)#passing embeddings through lstm
            last_output=output[:,-1,:]
            dropped=self.dropout(last_output)
            return self.fc(dropped)
        except Exception as e:
            logging.error(f"ERROR:Failed during model forwarding.")
            return torch.zeros(x.shape[0],1)
try:
    logging.info("Setting up model parameters and optimizers")
    vocab_size=len(vocab)
    embedded_dim=64
    hidden_dim=128
    output_dim=1
    dropout_prob=0.3
    epochs=40
    learning_rate=0.0005
    weight_decay=1*(10**-5)
    model=SpamHamModel(vocab_size,embedded_dim,hidden_dim,output_dim,dropout_prob)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer=optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay) 
    logging.info("Model,loss function and optimizer initialized")
except Exception as e:
    logging.error(f"ERROR:Failed during model set up as :{e}")
    exit()
class EarlyStopping:
    def __init__(self,patience):
        try: 
          self.patience=patience
          self.counter=0
          self.best_loss=float('inf')
          self.early_stop=False
          logging.info(f"Earlystopping initialized with patience:{patience}")
        except Exception as e:
            logging.error(f"ERROR:Failed to initialize for EarlyStopping:{e}")
    def __call__(self, val_loss):
        try:
             if val_loss< self.best_loss:
                  self.best_loss=val_loss
                  self.counter=0
                  logging.debug(f"EarlyStopping:New best validation loss:{val_loss:.4f}")
             else:
                 self.counter+=1
                 logging.info(f"EarlyStopping:Validation loss increases.Counter:{self.counter}/{self.patience}")
             return self.early_stop
        except Exception as e:
            logging.error(f"ERROR:Failed during earlystopping call:{e}")
            return False
early_stopper=EarlyStopping(patience=2)
epochs = 40
best_val_loss=float('inf')#initialize best validation loss to infinity
best_model_state=None
best_epoch=-1
logging.info(f"Starting training for {epochs} epochs.")
for epoch in range(epochs):
    try:   
      model.train()
      total_train_loss = 0
      for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x).squeeze(-1)
        loss = loss_fn(output, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()
      model.eval()
      total_val_loss = 0
      with torch.no_grad():
        for val_x, val_y in val_loader:
            val_output = model(val_x).squeeze(-1)
            val_loss = loss_fn(val_output, val_y)
            total_val_loss += val_loss.item()
      avg_train_loss=total_train_loss/len(train_loader)
      avg_val_loss = total_val_loss / len(val_loader)
      logging.info(f"Epoch {epoch + 1}/{epochs},Train Loss:{avg_train_loss:.4f},Val Loss:{avg_val_loss:.4f}")
      print(f"Epoch {epoch + 1}/{epochs},Train Loss:{avg_train_loss},Val Loss:{avg_val_loss}")
      if avg_val_loss < best_val_loss:
         best_val_loss=avg_val_loss
         best_epoch=epoch
         best_model_state=model.state_dict()
         logging.info(f"New best model found at epoch {epoch + 1}with Val Loss{avg_val_loss:.4f}")
         print(f"New best model found at epoch {epoch} with val loss {avg_val_loss:.4f}")
      if early_stopper(avg_val_loss):
       logging.info(f"ERROR:Early Stopping triggered at epoch {epoch + 1.}")
       print(f"Early stopping triggered at epoch {epoch+1}.")
       break
    except Exception as e:
        logging.error(f"ERROR:An error occured during epoch {epoch+1}:{e}")
try:
  if best_model_state is  not None:
      torch.save(best_model_state,"best_spamham_model.pth")
      logging.info(f"Best model saved from epoch {epoch +1}with val loss:{avg_val_loss:.4f}")
      print(f"Best model was saved from epoch {best_epoch} with val loss {best_val_loss:.4f}")  
  else:
     logging.warning("No model was saved because no improvement was found.")
     print("No model was saved because no improvement was found.")
except Exception as e:
    logging.error(f"ERROR:Failed to save the best model:{e}")
try:  
    with open("vocab.json","w")as file:
       json.dump(vocab,file)
    logging.info("Vocabulary sved to 'vocab.json'")
except Exception as e:
    logging.error(f"ERROR:Failed to save vocabulary to 'vocab.json'")
try: 
  model.load_state_dict(torch.load("best_spamham_model.pth"))
  model.eval()
  logging.info("Best model  loaded successfully for prediction.")
except FileNotFoundError:
    logging.critical(f"CRITICAL ERROR:'best_spamham_model.pth' not found.Cannot proceed")
    exit()
def predict(text):
    try:
        tokens=tokenize(text)
        encoded=encode_tokens(tokens,vocab)
        max_len=50       
        input_tensor=torch.tensor([encoded],dtype=torch.long)
        model.eval()
        with torch.no_grad():
           output=model(input_tensor).squeeze(-1)
           prob=torch.sigmoid(output).item()
           label="spam" if prob > 0.5 else "ham"
           logging.info(f"Prediction for '{text[:50]}....':{label.upper()}(confidence:{prob:.4f})")
           return label,prob
    except Exception as e:
        logging.error(f"ERROR:Failed to make predictions for'{text[:30]}.....:{e}")
        return "error",0.0
print("\nEnter mail messages to classify (type 'exit' to quit)")
logging.info("Starting user interaction loop for prediction")
while True:
     try:
       user_input=input(">>")
       if user_input.lower()=="exit":
          logging.info("User requested to exit the prediction loop.")
          break
       label,prob=predict(user_input)
       print(f"Prediction: {label.upper()} (confidence:{prob:.4f}\n)")
     except EOFError:
         logging.warning("Warning:EOFError encountered,possibly input stream closed.")
         break
     except Exception as e :
         logging.error(f"ERROR:An error occured during user  input or prediction:{e}")
         print("An error occured.Please try again .")
logging.info("Application Finished")
    
    


      










    

            

    



       
