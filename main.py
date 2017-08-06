import numpy as np
import sys
import os.path

new_model = True

# default je 1.0, bira output nasumicno
# nize vrednosti su konzervativne i obicno ponavljaju tekst koji je vidjen prilikom treniranja (tipa 0.7)
# vise vrednosti su rizicnije jer mogu sadrzati vise nepravilnosti (tipa 1.3)
sample_temperature = 1.0

# ulazni podaci
inpath = 'input.txt'

# izlazni podaci
outpath = 'out-' + inpath
smp_path = 'smp-' + inpath
dat_path = 'dat-' + inpath + '.npz'

# IO operacije
data = open(inpath, 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ("# podaci imaju {d} karaktera, {u} jedinstvenih".format(d = data_size, u = vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# osnovni hiperparametri
learning_rate = 0.1
learning_rate_decay = 0.97
learning_rate_decay_after = 10  # u epohama, kada krece smanjenje learning rate-a

# ukoliko pravimo nov model ili nemamo sacuvan checkpoint od ranije
if new_model or not os.path.isfile(dat_path):
    # hiperparametri modela
    hidden_size = 100  # broj neurona u hidden sloju
    seq_length = 25  # broj koraka

    # nasumicni paramteri modela
    Mih = np.random.randn(hidden_size, vocab_size) * 0.01  # input u hidden
    Mhh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden u hidden
    Mho = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden u output
    bh = np.zeros((hidden_size, 1))  # hidden bias
    bo = np.zeros((vocab_size, 1))  # output bias

    # ostali promenljivi parametri
    n, p, ni, epoch, p2 = 0, 0, 0, 0, 1

    # trenutno stanje u memoriji
    mMih, mMhh, mMho = np.zeros_like(Mih), np.zeros_like(Mhh), np.zeros_like(Mho)
    mbh, mbo = np.zeros_like(bh), np.zeros_like(bo)

    smooth_loss = -np.log(1.0 / vocab_size) * seq_length
else:
    loaded = np.load(dat_path)

    # hiperparametri modela
    hidden_size = loaded['hidden_size']
    seq_length = loaded['seq_length']

    # nasumicni paramteri modela
    Mih = loaded['Mih']
    Mhh = loaded['Mhh']
    Mho = loaded['Mho']
    bh = loaded['bh']
    bo = loaded['bo']

    # ostali promenljivi parametri
    n, p, ni, epoch, p2 = loaded['n'], loaded['p'], loaded['ni'], loaded['epoch'], loaded['p2']

    # trenutno stanje u memoriji
    mMih, mMhh, mMho = loaded['mMih'], loaded['mMhh'], loaded['mMho']
    mbh, mbo = loaded['mbh'], loaded['mbo']

    smooth_loss = loaded['smooth_loss']

    # memorija RNN mreze
    hprev = loaded['hprev']

    print ("# ucitano stanje iz {f}, epoha {e}, iteracija {n}".format(f = dat_path, e = epoch, n = n))
    if epoch > learning_rate_decay_after:
        learning_rate = (learning_rate * pow(learning_rate_decay, epoch - learning_rate_decay_after))

# standardna implementacija softmax funkcije (en.wikipedia.org/wiki/Softmax_function)
def softmax(w, temp=1.0):
    e = np.exp(np.array(w)/temp)
    dist = e / np.sum(e)
    return dist

# optimizovana softmax funkcija za temp = 1.0 (https://gist.github.com/stober/1946926)
def softmax_1(w):
    e = np.exp(w - np.max(w))
    dist = e / np.sum(e)
    return dist

# loss funkcija za RNN mrezu
def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward prolaz
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1)) # enkodiraj u 1-od-k reprezentaciji
        xs[t][inputs[t]] = 1
        # update-uj hidden sloj
        hs[t] = np.tanh(np.dot(Mih, xs[t]) + np.dot(Mhh, hs[t-1]) + bh)
        # izracunaj otuput vektor (nenormalnizovane log verovatnoce za sledece karaktere)
        ys[t] = np.dot(Mho, hs[t]) + bo
        ps[t] = softmax_1(ys[t]) # softmax verovatnoce za sledece karaktere
        loss += -np.log(ps[t][targets[t], 0]) # softmax (cross-entropy loss)
    # backward prolaz: racunamo gradijente prolazeci unazad
    dMih, dMhh, dMho = np.zeros_like(Mih), np.zeros_like(Mhh), np.zeros_like(Mho)
    dbh, dbo = np.zeros_like(bh), np.zeros_like(bo)

    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # backprop u y
        dMho += np.dot(dy, hs[t].T)
        dbo += dy
        dh = np.dot(Mho.T, dy) + dhnext # backprop u h
        dhraw = (1 - hs[t] * hs[t]) * dh # backprop kroz tanh nelinearnost
        dbh += dhraw
        dMih += np.dot(dhraw, xs[t].T)
        dMhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Mhh.T, dhraw)
    for dparam in [dMih, dMhh, dMho, dbh, dbo]:
        np.clip(dparam, -5, 5, out = dparam) # mitigacija za eksplodirajuce gradijente
    return loss, dMih, dMhh, dMho, dbh, dbo, hs[len(inputs) - 1]

# sample sekvenca brojeva iz modela
def sample(h, seed_ix, n):
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Mih, x) + np.dot(Mhh, h) + bh)
        y = np.dot(Mho, h) + bo
        p = softmax(y, sample_temperature)
        ix = np.random.choice(range(vocab_size), p = p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes

# blok teksta
def outblock(epoch, iter, loss, nchr):
    sample_ix = sample(hprev, inputs[0], nchr)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    output = ("# epoha {e}, iteracija {i}, loss {l:7.3f}, temp {t:5.3f}\n{txt}".format(e = epoch, i = iter, l = float(loss), t = sample_temperature, txt = txt))
    return output

output = ("# start\n# epoha {e}, iteracija {n}".format(e = epoch, n = n))
print (output)
f = open(outpath, 'a')
f.write(output + '\n')
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

    inputs = [char_to_ix[ch] for ch in data[p : p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1 : p + seq_length + 1]]

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
    loss, dMih, dMhh, dMho, dbh, dbo, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
  
    # adagrad update parametara
    for param, dparam, mem in zip([Mih,  Mhh,  Mho,  bh,  bo],
                                [dMih, dMhh, dMho, dbh, dbo],
                                [mMih, mMhh, mMho, mbh, mbo]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

    p += seq_length # data pointer
    n += 1 # broj iteracija

    if n > 0 and n % 4096 == 0:
        np.savez_compressed(dat_path,
            hidden_size = hidden_size, seq_length = seq_length,
            Mih = Mih, Mhh = Mhh, Mho = Mho, bh = bh, bo = bo,
            n = n, p = p, ni = ni, epoch = epoch, p2 = p2,
            mMih = mMih, mMhh = mMhh, mMho = mMho, mbh = mbh, mbo = mbo,
            smooth_loss=smooth_loss, hprev=hprev)

        print ("# sacuvano stanje u {f}, epoha {e}, iteracija {n}".format(f = dat_path, e = epoch, n = n))