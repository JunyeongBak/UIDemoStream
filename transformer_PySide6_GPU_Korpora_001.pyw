import sys
import torch
import torch.nn as nn
import math
import os
import xml.etree.ElementTree as ET
import re
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, 
                               QWidget, QTabWidget, QLineEdit, QFileDialog, QHBoxLayout, 
                               QLabel, QSpinBox)
from PySide6.QtCore import QThread, QObject, Signal, Slot

# 이 코드는 이전의 Transformer 실습을 실제 데이터로 학습시켜 성능을 개선합니다.
# [수정] Korpora 라이브러리의 데이터 로딩 불안정성을 해결하기 위해,
# 다운로드된 TMX 파일을 직접 파싱하도록 로직을 변경했습니다.
# ※ Korpora 라이브러리 설치가 필요합니다: pip install Korpora

# --- 데이터 처리 및 Vocabulary 클래스 ---
class Vocabulary:
    """단어와 인덱스를 매핑하는 Vocabulary 클래스"""
    def __init__(self, counter, specials=['<unk>', '<pad>', '<bos>', '<eos>']):
        self.specials = specials
        self.word2idx = {word: i for i, word in enumerate(specials)}
        for i, (word, _) in enumerate(counter.most_common(), len(specials)):
            self.word2idx[word] = i
        self.idx2word = {i: word for word, i in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)

def build_vocab(text_iter, tokenizer):
    """데이터로부터 Vocabulary를 생성하는 함수"""
    counter = Counter()
    for text in text_iter:
        counter.update(tokenizer(text))
    return Vocabulary(counter)

def data_process(text_pairs, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, device):
    """텍스트 데이터를 텐서로 변환하는 함수"""
    data = []
    for src_text, tgt_text in text_pairs:
        src_tensor = torch.tensor([src_vocab.word2idx.get(token, src_vocab.word2idx['<unk>']) 
                                   for token in src_tokenizer(src_text)], dtype=torch.long)
        tgt_tensor = torch.tensor([tgt_vocab.word2idx.get(token, tgt_vocab.word2idx['<unk>']) 
                                   for token in tgt_tokenizer(tgt_text)], dtype=torch.long)
        processed_src = torch.cat([torch.tensor([src_vocab.word2idx['<bos>']]), 
                                   src_tensor, 
                                   torch.tensor([src_vocab.word2idx['<eos>']])], dim=0)
        processed_tgt = torch.cat([torch.tensor([tgt_vocab.word2idx['<bos>']]), 
                                   tgt_tensor, 
                                   torch.tensor([tgt_vocab.word2idx['<eos>']])], dim=0)
        data.append((processed_src.to(device), processed_tgt.to(device)))
    return data

# --- Transformer 모델 코드 (이전과 동일) ---
class Transformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, 
                 dim_feedforward: int, dropout: float, src_vocab_size: int, tgt_vocab_size: int):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                tgt_mask: torch.Tensor, src_padding_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor, memory_key_padding_mask: torch.Tensor) -> torch.Tensor:
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask, 
                                          tgt_key_padding_mask=tgt_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
        return self.fc_out(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# --- PySide6 GUI 및 스레딩 코드 ---

class TrainingWorker(QObject):
    log_message = Signal(str)
    training_finished = Signal(object, object, object) # model, src_vocab, tgt_vocab

    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs

    @Slot()
    def run_training(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_message.emit(f"Using device: {device}")

            try:
                from Korpora import Korpora
                self.log_message.emit("Fetching OpenSubtitles dataset... (This might take a while on the first run)")
                Korpora.fetch("open_subtitles")

                korpora_path = os.path.join(os.path.expanduser('~'), 'Korpora', 'open_subtitles')
                tmx_path = os.path.join(korpora_path, 'en-ko.tmx')
                
                self.log_message.emit(f"Parsing TMX file from: {tmx_path}")
                tree = ET.parse(tmx_path)
                root = tree.getroot()
                TRAIN_DATA = []
                num_samples = 1000
                
                for tu in root.iter('tu'):
                    if len(TRAIN_DATA) >= num_samples:
                        break
                    
                    en_text, ko_text = None, None
                    for tuv in tu.findall('tuv'):
                        lang = tuv.get('{http://www.w3.org/XML/1998/namespace}lang')
                        seg = tuv.find('seg')
                        # [FIX] 데이터 정제 로직 추가
                        if seg is not None and seg.text:
                            # 문장 앞뒤 공백 및 `-` 기호 제거
                            cleaned_text = re.sub(r'^\s*-\s*', '', seg.text).strip()
                            if not cleaned_text:
                                continue # 정제 후 내용이 없으면 건너뜀

                            if lang == 'en':
                                en_text = cleaned_text.lower()
                            elif lang == 'ko':
                                ko_text = cleaned_text
                    
                    if en_text and ko_text:
                        TRAIN_DATA.append((en_text, ko_text))
                
                self.log_message.emit(f"Successfully loaded and cleaned {len(TRAIN_DATA)} samples.")

            except Exception as e:
                self.log_message.emit(f"Korpora download or parsing failed: {e}")
                self.log_message.emit("Using a built-in sample dataset instead.")
                TRAIN_DATA = [
                    ("i am a student", "나는 학생 입니다"),
                    ("this is a book", "이것은 책 입니다"),
                    ("i love this city", "나는 이 도시 를 사랑 합니다"),
                    ("he is a teacher", "그는 선생님 입니다"),
                    ("she reads a book", "그녀는 책 을 읽습니다"),
                    ("the weather is nice today", "오늘 날씨 가 좋습니다"),
                    ("what is your name", "당신의 이름 은 무엇입니까"),
                    ("they are playing soccer", "그들은 축구 를 하고 있습니다"),
                    ("this is my new car", "이것은 나의 새 차 입니다"),
                    ("the cat is sleeping on the sofa", "고양이 가 소파 에서 자고 있습니다"),
                    ("please open the door", "문 을 열어 주세요"),
                    ("thank you very much", "정말 감사 합니다")
                ]

            src_tokenizer = str.split
            tgt_tokenizer = str.split

            src_vocab = build_vocab([pair[0] for pair in TRAIN_DATA], src_tokenizer)
            tgt_vocab = build_vocab([pair[1] for pair in TRAIN_DATA], tgt_tokenizer)
            self.log_message.emit(f"Source vocab size: {len(src_vocab)}")
            self.log_message.emit(f"Target vocab size: {len(tgt_vocab)}")

            SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = len(src_vocab), len(tgt_vocab)
            D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD = 128, 4, 2, 2, 512
            DROPOUT = 0.1
            
            model = Transformer(d_model=D_MODEL, nhead=NHEAD, num_encoder_layers=NUM_ENCODER_LAYERS,
                                num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                                dropout=DROPOUT,
                                src_vocab_size=SRC_VOCAB_SIZE, tgt_vocab_size=TGT_VOCAB_SIZE).to(device)
            
            PAD_IDX = src_vocab.word2idx['<pad>']
            criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            self.log_message.emit("\n--- Start Training Loop ---")
            model.train()
            
            BATCH_SIZE = 16
            processed_data = data_process(TRAIN_DATA, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, device)

            for epoch in range(self.epochs):
                total_loss = 0
                batch_count = 0
                for i in range(0, len(processed_data), BATCH_SIZE):
                    batch = processed_data[i:i + BATCH_SIZE]
                    src_list = [p[0] for p in batch]
                    tgt_list = [p[1] for p in batch]

                    src_batch = pad_sequence(src_list, batch_first=True, padding_value=PAD_IDX)
                    tgt_batch = pad_sequence(tgt_list, batch_first=True, padding_value=PAD_IDX)

                    tgt_input = tgt_batch[:, :-1]
                    tgt_output = tgt_batch[:, 1:]

                    src_padding_mask = (src_batch == PAD_IDX)
                    tgt_padding_mask = (tgt_input == PAD_IDX)
                    tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device)

                    optimizer.zero_grad()
                    output = model(src_batch, tgt_input, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                    loss = criterion(output.contiguous().view(-1, TGT_VOCAB_SIZE), tgt_output.contiguous().view(-1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batch_count +=1
                
                avg_loss = total_loss / batch_count
                self.log_message.emit(f"Epoch: {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

            self.log_message.emit("\n--- Training Finished ---")
            self.training_finished.emit(model, src_vocab, tgt_vocab)

        except Exception as e:
            self.log_message.emit(f"\nAn error occurred: {e}")
            self.training_finished.emit(None, None, None)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Transformer Model GUI (Korpora Ver.)")
        self.setGeometry(100, 100, 700, 600)

        self.model = None
        self.src_vocab = None
        self.tgt_vocab = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tabs = QTabWidget()
        self.train_tab = QWidget()
        self.inference_tab = QWidget()
        self.tabs.addTab(self.train_tab, "Model Training")
        self.tabs.addTab(self.inference_tab, "Inference")
        
        train_layout = QVBoxLayout(self.train_tab)
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("Epochs:"))
        self.epoch_spinbox = QSpinBox()
        self.epoch_spinbox.setRange(1, 1000); self.epoch_spinbox.setValue(100)
        epoch_layout.addWidget(self.epoch_spinbox)
        self.start_button = QPushButton("Start Training")
        self.save_button = QPushButton("Save Trained Model"); self.save_button.setEnabled(False)
        train_layout.addLayout(epoch_layout); train_layout.addWidget(self.start_button); train_layout.addWidget(self.save_button)
        
        inference_layout = QVBoxLayout(self.inference_tab)
        self.load_button = QPushButton("Load Model from .pkl")
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input Sentence (English):"))
        self.input_line = QLineEdit("i am a student")
        input_layout.addWidget(self.input_line)
        self.inference_button = QPushButton("Translate"); self.inference_button.setEnabled(False)
        inference_layout.addWidget(self.load_button); inference_layout.addLayout(input_layout); inference_layout.addWidget(self.inference_button)

        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        main_layout = QVBoxLayout(); main_layout.addWidget(self.tabs); main_layout.addWidget(QLabel("Logs:")); main_layout.addWidget(self.log_box)
        container = QWidget(); container.setLayout(main_layout); self.setCentralWidget(container)

        self.start_button.clicked.connect(self.run_model_training)
        self.save_button.clicked.connect(self.save_model)
        self.load_button.clicked.connect(self.load_model)
        self.inference_button.clicked.connect(self.run_inference)

    def run_model_training(self):
        self.start_button.setEnabled(False); self.save_button.setEnabled(False)
        self.log_box.clear(); self.log_box.append("Starting training process...")
        epochs = self.epoch_spinbox.value()
        self.thread = QThread(); self.worker = TrainingWorker(epochs); self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run_training)
        self.worker.training_finished.connect(self.on_training_finished)
        self.worker.log_message.connect(self.log_box.append)
        self.thread.start()

    def on_training_finished(self, model, src_vocab, tgt_vocab):
        self.model, self.src_vocab, self.tgt_vocab = model, src_vocab, tgt_vocab
        self.start_button.setEnabled(True)
        if self.model:
            self.save_button.setEnabled(True); self.inference_button.setEnabled(True)
            self.log_box.append("Model is trained. Ready to save or inference.")
        else:
            self.log_box.append("Training failed.")
        self.thread.quit(); self.thread.wait(); self.thread.deleteLater(); self.worker.deleteLater()

    def save_model(self):
        if not self.model:
            self.log_box.append("No trained model to save."); return
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "Pickle Files (*.pkl)")
        if filePath:
            try:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'src_vocab': self.src_vocab,
                    'tgt_vocab': self.tgt_vocab,
                }, filePath)
                self.log_box.append(f"Model and vocab saved to: {filePath}")
            except Exception as e:
                self.log_box.append(f"Error saving model: {e}")

    def load_model(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Pickle Files (*.pkl)")
        if filePath:
            try:
                checkpoint = torch.load(filePath, map_location=self.device, weights_only=False)
                self.src_vocab = checkpoint['src_vocab']
                self.tgt_vocab = checkpoint['tgt_vocab']
                
                self.model = Transformer(d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                                         dim_feedforward=512, dropout=0.1, 
                                         src_vocab_size=len(self.src_vocab), tgt_vocab_size=len(self.tgt_vocab))
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                self.inference_button.setEnabled(True)
                self.log_box.append(f"Model and vocab loaded successfully from: {filePath}")
            except Exception as e:
                self.log_box.append(f"Error loading model: {e}")

    def run_inference(self):
        if not self.model:
            self.log_box.append("No model loaded for inference."); return
        
        self.model.eval()
        input_text = self.input_line.text().lower()
        src_tokenizer = str.split
        
        tokens = [self.src_vocab.word2idx.get(t, self.src_vocab.word2idx['<unk>']) for t in src_tokenizer(input_text)]
        src_tokens = [self.src_vocab.word2idx['<bos>']] + tokens + [self.src_vocab.word2idx['<eos>']]
        src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(self.device)
        src_padding_mask = (src_tensor == self.src_vocab.word2idx['<pad>']).to(self.device)

        memory = self.model.transformer_encoder(
            self.model.pos_encoder(self.model.src_embedding(src_tensor) * math.sqrt(self.model.d_model)),
            src_key_padding_mask=src_padding_mask
        )
        
        tgt_tokens = [self.tgt_vocab.word2idx['<bos>']]
        for i in range(50):
            tgt_tensor = torch.LongTensor(tgt_tokens).unsqueeze(0).to(self.device)
            tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(1), self.device)
            
            output = self.model.transformer_decoder(
                self.model.pos_encoder(self.model.tgt_embedding(tgt_tensor) * math.sqrt(self.model.d_model)),
                memory,
                tgt_mask=tgt_mask
            )
            pred_token = self.model.fc_out(output[:, -1, :]).argmax(1).item()
            tgt_tokens.append(pred_token)
            if pred_token == self.tgt_vocab.word2idx['<eos>']:
                break
        
        translated_tokens = [self.tgt_vocab.idx2word[i] for i in tgt_tokens]
        self.log_box.append(f"\n--- Inference Result ---")
        self.log_box.append(f"Input: '{input_text}'")
        self.log_box.append(f"Translated: {' '.join(translated_tokens[1:-1])}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
