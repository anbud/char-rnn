import numpy as np
import sys
import re
import os.path

new_model = False

# default je 1.0, bira output nasumicno
# nize vrednosti su konzervativne i obicno ponavljaju tekst koji je vidjen prilikom treniranja (tipa 0.7)
# vise vrednosti su rizicnije jer mogu sadrzati vise nepravilnosti (tipa 1.3)
sample_temperature = 1.0

# ulazni podaci
inpath = 'input.txt'

# izlazni podaci
outpath = 'out-' + inpath
smp_path = 'smp-' + inpath
dat_path = 'dat-' + inpath

dat_path = re.sub(r'\.txt$', '.npz', dat_path)

# data I/O
data = open(inpath, 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ("# podaci imaju {d} karaktera, {u} jedinstvenih".format(d = data_size, u = vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

learning_rate = 0.1
learning_rate_decay = 0.97
learning_rate_decay_after = 10  # u epohama, kada krece smanjenje learning rate-a

if new_model or not os.path.isfile(dat_path):
    # hiperparametri modela
    hidden_size = 100  # broj neurona u hidden sloju
    seq_length = 25  # broj koraka

    # nasumicni paramteri modela
    W_xh = np.random.randn(hidden_size, vocab_size) * 0.01  # input u hidden
    W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden u hidden
    W_hy = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden u output
    bh = np.zeros((hidden_size, 1))  # hidden bias
    by = np.zeros((vocab_size, 1))  # output bias

    # ostali promenljivi parametri
    n, p = 0, 0
    ni = 0
    epoch = 0
    p2 = 1
    mW_xh, mW_hh, mW_hy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by)
    smooth_loss = -np.log(1.0 / vocab_size) * seq_length
else:
    loaded = np.load(dat_path)
    hidden_size = loaded['hidden_size']
    seq_length = loaded['seq_length']
    W_xh = loaded['W_xh']
    W_hh = loaded['W_hh']
    W_hy = loaded['W_hy']
    bh = loaded['bh']
    by = loaded['by']
    n = loaded['n']
    p = loaded['p']
    ni = loaded['ni']
    epoch = loaded['epoch']
    p2 = loaded['p2']
    mW_xh = loaded['mW_xh']
    mW_hh = loaded['mW_hh']
    mW_hy = loaded['mW_hy']
    mbh = loaded['mbh']
    mby = loaded['mby']
    smooth_loss = loaded['smooth_loss']
    hprev = loaded['hprev']
    print ("# ucitano stanje iz {f}, epoha {e}, iteracija {n}".format(f = dat_path, e = epoch, n = n))
    if epoch > learning_rate_decay_after:
        learning_rate = (learning_rate * pow(learning_rate_decay, epoch-learning_rate_decay_after))

def softmax(w, temp=1.0):
    """
    standardna implementacija softmax funkcije (en.wikipedia.org/wiki/Softmax_function)
    """
    e = np.exp(np.array(w)/temp)
    dist = e / np.sum(e)
    return dist

def softmax_1(x):
    """
    optimizovana verzija kada je temp=1.0
    source: gist.github.com/stober/1946926
    """
    e_x = np.exp(x - np.max(x))
    out = e_x / np.sum(e_x)
    return out

def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        # update the hidden state
        hs[t] = np.tanh(np.dot(W_xh, xs[t]) + np.dot(W_hh, hs[t-1]) + bh)
        # compute the output vector (unnormalized log prob'ys for next chars)
        ys[t] = np.dot(W_hy, hs[t]) + by
        ps[t] = softmax_1(ys[t]) # prob'ys for next chars
        loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    # backward pass: compute gradients going backwards
    dW_xh = np.zeros_like(W_xh)
    dW_hh, dW_hy = np.zeros_like(W_hh), np.zeros_like(W_hy)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # backprop into y
        dW_hy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(W_hy.T, dy) + dhnext # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
        dbh += dhraw
        dW_xh += np.dot(dhraw, xs[t].T)
        dW_hh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(W_hh.T, dhraw)
    for dparam in [dW_xh, dW_hh, dW_hy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # mitigate exploding gradients
    return loss, dW_xh, dW_hh, dW_hy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
    """ 
    sample sekvenca broja iz modela
    h je memorijsko stanje, seed_ix je seed slovo, za pocetak sample-a
    """
    x = np.zeros((vocab_size, 1)) ; x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(W_xh, x) + np.dot(W_hh, h) + bh)
        y = np.dot(W_hy, h) + by
        # For each possible next letter, the probability that we'll choose
        # that letter is proportional to the output value we just computed;
        # and the sum of all probs must be 1.0. This is the "Softmax"
        # calculation.
        p = softmax(y, sample_temperature)
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1)) ; x[ix] = 1
        ixes.append(ix)
    return ixes

def outblock(epoch, iter, loss, nchr):
    """
    blok teksta
    """
    sample_ix = sample(hprev, inputs[0], nchr)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    outpt = ("# epoha {e}, iteracija {i}, loss {l:7.3f}, temp {t:5.3f}\n{txt}".format(e = epoch, i = iter, l = float(loss), t = sample_temperature, txt = txt))
    return outpt

outpt = ("# start\n# epoha {e}, iteracija {n}".format(e=epoch, n=n))
print (outpt)
f = open(outpath, 'a')
f.write(outpt + '\n')
f.close()

while True:
  # priprema ulaznih podataka
  if p + seq_length + 1 >= len(data) or n == 0:
    # nemamo dovoljno podataka za sledeci seq_length batch, krecemo u novu epohu
    hprev = np.zeros((hidden_size, 1)) # resetuj RNN memoriju
    epoch += 1
    p = 0

    if epoch > learning_rate_decay_after:
      learning_rate = (learning_rate * pow(learning_rate_decay, epoch - learning_rate_decay_after))
      if learning_rate > 0.05:
        print ("# epoha {e}, smanji learning rate za {lrd:4.2f} do {lr:5.3f}".format(e = epoch, lrd = learning_rate_decay, lr = learning_rate))

  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  if n > 0 and n % 100 == 0:
    ni = ni + 1
    if ni >= p2:
      outpt = outblock(epoch, n, smooth_loss, 200)
      print (outpt)
      outpt = outblock(epoch, n, smooth_loss, 1024)
      f = open(outpath, 'a')
      f.write(outpt + '\n')
      f.close()
      p2 = p2 * 2

  if n > 0 and n % 4096 == 0:
    outpt = outblock(epoch, n, smooth_loss, 4096)
    f = open(smp_path, 'w')
    f.write(outpt)
    f.close()

  # forward seq_length karaktera kroz mrezu i uzmi gradijent
  loss, dW_xh, dW_hh, dW_hy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  
  # adagrad update parametara
  for param, dparam, mem in zip([ W_xh,  W_hh,  W_hy,  bh,  by], 
                                [dW_xh, dW_hh, dW_hy, dbh, dby], 
                                [mW_xh, mW_hh, mW_hy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # data pointer
  n += 1 # broj iteracija

  if n > 0 and n % 4096 == 0:
    np.savez_compressed(dat_path,
      hidden_size=hidden_size, seq_length=seq_length,
      W_xh=W_xh, W_hh=W_hh, W_hy=W_hy, bh=bh, by=by,
      n=n, p=p, ni=ni, epoch=epoch, p2=p2,
      mW_xh=mW_xh, mW_hh=mW_hh, mW_hy=mW_hy, mbh=mbh, mby=mby,
      smooth_loss=smooth_loss, hprev=hprev)
    print ("# sacuvano stanje u {f}, epoha {e}, iteracija {n}".format(f = dat_path, e = epoch, n = n))