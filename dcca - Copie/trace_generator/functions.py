from requirements import *


def montgomery_mult(m, n, d):
    r_0, r_1 = 1, 1
    r = [r_0, r_1]
    for i in bin(d)[2:]:
        d_j = 1 - int(i)
        r[0] = (r[0] * r[0]) % n
        r[d_j] = (r[0] * m) % n
    return r[0]

def compress(t_ij):
    t_x = [[] for _ in range(len(t_ij))]
    final_t_x = []
    for i in range(len(t_ij)):
        sum = 0
        for j in range(len(t_ij[i])):
            sum += t_ij[i][j]
        t_x[i].append((1/(len(t_ij[i]))) *sum)
    for i in range(len(t_x)):
        final_t_x.append(t_x[i][0])
    return final_t_x

def correlation(t_x,t_x_prime):
    return np.corrcoef(t_x, t_x_prime)[0][1]

def plot_traces(compressed_t):
    final_trace = []
    for i in range(len(compressed_t)):
        final_trace.append(compressed_t[i])
    plt.plot(final_trace)
    plt.show()



def split_int_to_words(a,word_size):
    if len(bin(a)[2:]) % word_size == 0:
        words = [[] for _ in range((len(bin(a)[2:])//word_size))]
        temp = 0
        for k in range((len(bin(a)[2:])//word_size)):
            for i in range(word_size):
                words[k].append(bin(a)[2:][i+temp])
            temp += word_size
    else:
        words = [[] for _ in range((len(bin(a)[2:])//word_size)+1)]
        temp = 0
        for k in range((len(bin(a)[2:])//word_size)+1):
            if k != (len(bin(a)[2:])//word_size):
                for i in range(word_size):
                    words[k].append(bin(a)[2:][i+temp])
                temp += word_size
            else:
                for i in range(len(bin(a)[2:])-temp):
                    words[k].append(bin(a)[2:][temp+i])
    final_words = []
    for i in range(len(words)):
        str = ''
        for k in words[i]:
            str += k
        final_words.append(str)

    return final_words


def words_to_int(splitted_words):
    list_ints = []
    splitted_words.reverse()
    for i in range(len(splitted_words)):
        splitted_words[i] = splitted_words[i][::-1]
        
    cmp = 0

    for i in range(len(splitted_words)):
        sum = 0
        for k in range(len(splitted_words[i])):

            sum += (2**(cmp))*int(splitted_words[i][k][::-1])
            cmp += 1
        list_ints.append(sum)

    return list_ints[::-1]


def LIM(x,y, wordsize):

    x = int(bin(x)[2:].zfill(2048), 2)
    y = int(bin(y)[2:].zfill(2048), 2)
    
    x_split = split_int_to_words(x, wordsize)
    list_ints_x = words_to_int(x_split)
    
    y_split = split_int_to_words(y, wordsize)
    list_ints_y = words_to_int(y_split)

    
    list_ints_x = list_ints_x[::-1]
    list_ints_y = list_ints_y[::-1]
    
    trace = [[] for _ in range(len(list_ints_x))]
    res = 0

    for i in range(len(list_ints_x)):
        for j in range(len(list_ints_y)):
            res += list_ints_x[i]*list_ints_y[j]
            trace[i].append(bin(list_ints_x[i]*list_ints_y[j]).count('1'))
    return res, trace


def square_multiply_always(m,n,d, wordsize):
    r0 = 1
    r1 = 1

    t_s = []
    t_m = []

    for i in tqdm(range(len(bin(d)[2:]))):
        temp = 0

        lim = LIM(r1,r1, wordsize)
        temp += (lim[0])
        t_s.append(lim[1])

        r1 = temp % n

        temp = 0

        if int(bin(d)[2:][i]) == 0:
            lim = LIM(r1,m, wordsize)

            r0 += lim[0] 
            t_m.append(lim[1])
            r0 = r0 % n
        else:
            lim = LIM(r1,m, wordsize)
            temp += lim[0]
            t_m.append(lim[1])
            r1 = temp % n
    return r1, t_s, t_m 

def add_noise(traces,mu,sigma):
    noised_traces = [[[] for _ in range(len(traces[i]))] for i in range(len(traces))]
    cmp = 0

    for trace in range(len(traces)):
        for j in range(len(traces[trace])):
            noise = np.random.normal(mu,sigma,(len(traces[trace])))
            noised_traces[trace][j] = traces[trace][j]+noise
        cmp += len(traces[trace])

    return noised_traces



    
        