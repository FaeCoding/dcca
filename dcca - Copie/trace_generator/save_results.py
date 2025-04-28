import sys
from pathlib import Path

# Get the parent directory path
parent_dir = str(Path(__file__).resolve().parent.parent)
# Add it to Python path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from requirements import *

from functions import *

with open(f'../traces/traces_square_mult_0.pkl', 'rb') as f:
        t_s, t_m = pickle.load(f)
with open(f'../variables/variables_0.pkl', 'rb') as f:
        m, n, d = pickle.load(f)

################################## RAW TRACES ##############################################
list_corr_brute = []
t_s_p = [[] for _ in range(len(t_s))]
t_m_p = [[] for _ in range(len(t_m))]
for i in range(1,len(t_s)):
    for j in range(len(t_s[i])):
        for k in range(len(t_s[i][j])):
            t_s_p[i].append(t_s[i][j][k])
            t_m_p[i].append(t_m[i][j][k])
for i in range(2,len(t_s)):
    list_corr_brute.append(np.corrcoef(t_s_p[i], t_m_p[i-1])[0][1])

med = np.median(list_corr_brute)
accu = 1
d_bits = bin(d)[2:]
for i in range(len(list_corr_brute)):
    if list_corr_brute[i] > med and int(d_bits[i+1]) == 0:
        accu += 1
    elif list_corr_brute[i] < med and int(d_bits[i+1]) == 1:
        accu += 1
# Separate correlation values based on the bit value
corr_bit1 = [list_corr_brute[i] for i in range(len(list_corr_brute)) if d_bits[i+1] == '1']
corr_bit0 = [list_corr_brute[i] for i in range(len(list_corr_brute)) if d_bits[i+1] == '0']
indices_bit1 = [i for i in range(len(list_corr_brute)) if d_bits[i+1] == '1']
indices_bit0 = [i for i in range(len(list_corr_brute)) if d_bits[i+1] == '0']
# Plot
plt.figure(figsize=(10, 6))
plt.plot(corr_bit1, indices_bit1, 'og', label='Bit=1')  # Green for bits=1
plt.plot(corr_bit0, indices_bit0, 'ob', label='Bit=0')  # Blue for bits=0
plt.axvline(x=med, color='r', label='Median')
plt.legend()
plt.xlabel("Correlation value")
plt.ylabel("Nb pairs")
plt.title(f"Collision Correlation Attack with ground-truth labels without Preprocess BigMac (Accuracy: {accu*100/len(d_bits):.2f}%)")
plt.savefig('results/raw_traces/raw_traces_MEDIANE.png', dpi=300, bbox_inches='tight')

# Reshape list_corr for K-means (requires 2D array)
X = np.array(list_corr_brute).reshape(-1, 1)
# Apply K-means (2 clusters)
kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
labels = kmeans.labels_
# Determine which cluster corresponds to 0/1 (lower mean = 0, higher mean = 1)
cluster_means = [X[labels == 0].mean(), X[labels == 1].mean()]
if cluster_means[0] < cluster_means[1]:
    labels = 1 - labels  # Swap labels if cluster order is reversed

# Calculate accuracy
accu_k = 1 # We start at 1 because we don't take into account the most significant bit = 1.
for i in range(len(labels)):
    if labels[i] == int(d_bits[i+1]) :
        accu_k += 1
accuracy = accu_k * 100 / len(d_bits)
a = [X[i] for i in range(len(X)) if d_bits[i+1] == '1']
b = [X[i] for i in range(len(X)) if d_bits[i+1] == '0']
a_prime = [i for i in range(len(list_corr_brute)) if d_bits[i+1] == '1']
b_prime = [i for i in range(len(list_corr_brute)) if d_bits[i+1] == '0']
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(a, a_prime, 'og', label='Bit=1')  # Green for bits=1
plt.plot(b, b_prime, 'ob', label='Bit=0')  # Blue for bits=0
plt.axvline(x=kmeans.cluster_centers_.mean(), color='r', linestyle='--', label='Cluster Boundary')
plt.legend()
plt.xlabel("Correlation Value")
plt.ylabel("Bit Position")
plt.title(f"K-means Clustering (Accuracy: {accuracy:.2f}%)")
plt.savefig('results/raw_traces/raw_traces_KMEANS.png', dpi=300, bbox_inches='tight')



################################## BIGMAC ##############################################
list_s_compressed = []
list_m_compressed = []
for i in range(len(t_s)):
    list_s_compressed.append(compress(t_s[i]))
    list_m_compressed.append(compress(t_m[i]))
list_corr = []
for i in range(2,len(list_s_compressed)):
    list_corr.append(np.corrcoef(list_s_compressed[i], list_m_compressed[i-1])[0][1])
    med = np.median(list_corr)
accu = 1
d_bits = bin(d)[2:]
for i in range(len(list_corr)):
    if list_corr[i] > med and int(d_bits[i+1]) == 0:
        accu += 1
    elif list_corr[i] < med and int(d_bits[i+1]) == 1:
        accu += 1
# Separate correlation values based on the bit value
corr_bit1 = [list_corr[i] for i in range(len(list_corr)) if d_bits[i+1] == '1']
corr_bit0 = [list_corr[i] for i in range(len(list_corr)) if d_bits[i+1] == '0']
indices_bit1 = [i for i in range(len(list_corr)) if d_bits[i+1] == '1']
indices_bit0 = [i for i in range(len(list_corr)) if d_bits[i+1] == '0']
# Plot
plt.figure(figsize=(10, 6))
plt.plot(corr_bit1, indices_bit1, 'og', label='Bit=1')  # Green for bits=1
plt.plot(corr_bit0, indices_bit0, 'ob', label='Bit=0')  # Blue for bits=0
plt.axvline(x=med, color='r', label='Median')
plt.legend()
plt.xlabel("Correlation value")
plt.ylabel("Nb pairs")
plt.title(f"Collision Correlation Attack with ground-truth labels with PreProcess BigMac (Accuracy: {accu*100/len(d_bits):.2f}%)")
plt.savefig('results/bigmac/n0_MEDIANE.png', dpi=300, bbox_inches='tight')

# Reshape list_corr for K-means (requires 2D array)
X = np.array(list_corr).reshape(-1, 1)
# Apply K-means (2 clusters)
kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
labels = kmeans.labels_
# Determine which cluster corresponds to 0/1 (lower mean = 0, higher mean = 1)
cluster_means = [X[labels == 0].mean(), X[labels == 1].mean()]
if cluster_means[0] < cluster_means[1]:
    labels = 1 - labels  # Swap labels if cluster order is reversed
# Calculate accuracy
accu_k = 1 # We start at 1 because we don't take into account the most significant bit = 1.
for i in range(len(labels)):
    if labels[i] == int(d_bits[i+1]) :
        accu_k += 1
accuracy = accu_k * 100 / len(d_bits)
a = [X[i] for i in range(len(X)) if d_bits[i+1] == '1']
b = [X[i] for i in range(len(X)) if d_bits[i+1] == '0']
a_prime = [i for i in range(len(list_corr)) if d_bits[i+1] == '1']
b_prime = [i for i in range(len(list_corr)) if d_bits[i+1] == '0']
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(a, a_prime, 'og', label='Bit=1')  # Green for bits=1
plt.plot(b, b_prime, 'ob', label='Bit=0')  # Blue for bits=0
plt.axvline(x=kmeans.cluster_centers_.mean(), color='r', linestyle='--', label='Cluster Boundary')
plt.legend()
plt.xlabel("Correlation Value")
plt.ylabel("Bit Position")
plt.title(f"K-means Clustering (Accuracy: {accuracy:.2f}%)")
plt.savefig('results/bigmac/n0_KMEANS.png', dpi=300, bbox_inches='tight')


##################################  BIGMAC + NOISE ##############################################
##################################  NOISE == 1   MEADIANE #######################################
noised_s = add_noise(t_s,0,1)
noised_m = add_noise(t_m,0,1)
list_s_compressed_n1 = []
list_m_compressed_n1 = []
for i in range(len(t_s)):
    list_s_compressed_n1.append(compress(noised_s[i]))
    list_m_compressed_n1.append(compress(noised_m[i]))
list_corr_n1 = []
for i in range(2,len(list_s_compressed_n1)):
    list_corr_n1.append(np.corrcoef(list_s_compressed_n1[i], list_m_compressed_n1[i-1])[0][1])
med_n3_n1 = np.median(list_corr_n1)
accu_n1 = 1 
for i in range(len(list_corr_n1)):
    if list_corr_n1[i] > med_n3_n1 and int(bin(d)[2:][i+1]) == 0:
        accu_n1 += 1
    elif list_corr_n1[i] < med_n3_n1 and int(bin(d)[2:][i+1]) == 1:
        accu_n1 += 1
# Separate correlation values based on the bit value
corr_bit1_n1 = [list_corr_n1[i] for i in range(len(list_corr_n1)) if d_bits[i+1] == '1']
corr_bit0_n1 = [list_corr_n1[i] for i in range(len(list_corr_n1)) if d_bits[i+1] == '0']
indices_bit1_n1 = [i for i in range(len(list_corr_n1)) if d_bits[i+1] == '1']
indices_bit0_n1 = [i for i in range(len(list_corr_n1)) if d_bits[i+1] == '0']
# Plot
plt.figure(figsize=(10, 6))
plt.plot(corr_bit1_n1, indices_bit1_n1, 'og', label='Bit=1')  # Green for bits=1
plt.plot(corr_bit0_n1, indices_bit0_n1, 'ob', label='Bit=0')  # Blue for bits=0
plt.axvline(x=med_n3_n1, color='r', label='median')
plt.legend()
plt.xlabel("Correlation value")
plt.ylabel("Nb pairs")
plt.title(f"Collision Correlation Attack with noise (Sigma = 1) (Accuracy: {accu_n1*100/len(d_bits):.2f}%)")
plt.savefig('results/bigmac/n1_MEDIANE.png', dpi=300, bbox_inches='tight')

##################################    NOISE == 1   KMEANS ##############################################
# Reshape list_corr for K-means (requires 2D array)
X_n1 = np.array(list_corr_n1).reshape(-1, 1)
kmeans_n1 = KMeans(n_clusters=2, random_state=42).fit(X_n1)
labels_n1 = kmeans_n1.labels_
# Determine which cluster corresponds to 0/1 (lower mean = 0, higher mean = 1)
cluster_means_n1 = [X_n1[labels == 0].mean(), X_n1[labels == 1].mean()]
if cluster_means_n1[0] < cluster_means_n1[1]:
    labels_n1 = 1 - labels_n1  # Swap labels if cluster order is reversed
# Calculate accuracy
accu_k_n1 = 1
for i in range(len(labels_n1)):
    if labels_n1[i] == int(d_bits[i+1]) :
        accu_k_n1 += 1
accuracy_n1 = accu_k_n1 * 100 / len(d_bits)
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(corr_bit1_n1, indices_bit1_n1, 'og', label='Bit=1')  # Green for bits=1
plt.plot(corr_bit0_n1, indices_bit0_n1, 'ob', label='Bit=0')  # Blue for bits=0
plt.axvline(x=kmeans_n1.cluster_centers_.mean(), color='r', linestyle='--', label='Cluster Boundary')
plt.legend()
plt.xlabel("Correlation Value")
plt.ylabel("Bit Position")
plt.title(f"K-means Clustering with Noise (Sigma = 1) (Accuracy: {accuracy_n1}%)")
plt.savefig('results/bigmac/n1_KMEANS.png', dpi=300, bbox_inches='tight')

##################################    NOISE == 2   MEADIANE ##############################################
noised_2_s = add_noise(t_s,0,2)
noised_2_m = add_noise(t_m,0,2)
list_s_compressed_n2 = []
list_m_compressed_n2 = []
for i in range(len(t_s)):
    list_s_compressed_n2.append(compress(noised_2_s[i]))
    list_m_compressed_n2.append(compress(noised_2_m[i]))
list_corr_n2 = []
for i in range(2,len(list_s_compressed_n2)):
    list_corr_n2.append(np.corrcoef(list_s_compressed_n2[i], list_m_compressed_n2[i-1])[0][1])
med_n2 = np.median(list_corr_n2)
accu_n2 = 1
for i in range(len(list_corr_n2)):
    if list_corr_n2[i] > med_n2 and int(bin(d)[2:][i+1]) == 0:
        accu_n2 += 1
    elif list_corr_n2[i] < med_n2 and int(bin(d)[2:][i+1]) == 1:
        accu_n2 += 1
# Separate correlation values based on the bit value
corr_bit1_n2 = [list_corr_n2[i] for i in range(len(list_corr_n2)) if d_bits[i+1] == '1']
corr_bit0_n2 = [list_corr_n2[i] for i in range(len(list_corr_n2)) if d_bits[i+1] == '0']
indices_bit1_n2 = [i for i in range(len(list_corr_n2)) if d_bits[i+1] == '1']
indices_bit0_n2 = [i for i in range(len(list_corr_n2)) if d_bits[i+1] == '0']
# Plot
plt.figure(figsize=(10, 6))
plt.plot(corr_bit1_n2, indices_bit1_n2, 'og', label='Bit=1')  # Green for bits=1
plt.plot(corr_bit0_n2, indices_bit0_n2, 'ob', label='Bit=0')  # Blue for bits=0
plt.axvline(x=med_n2, color='r', label='Median')
plt.legend()
plt.xlabel("Correlation value")
plt.ylabel("Nb pairs")
plt.title(f"Collision Correlation Attack with noise (Sigma = 2) (Accuracy: {accu_n2*100/len(d_bits):.2f}%)")
plt.savefig('results/bigmac/n2_MEDIANE.png', dpi=300, bbox_inches='tight')

##################################    NOISE == 2   KMEANS ##############################################
# Reshape list_corr for K-means (requires 2D array)
X_n2 = np.array(list_corr_n2).reshape(-1, 1)
kmeans_n2 = KMeans(n_clusters=2, random_state=42).fit(X_n2)
labels_n2 = kmeans_n2.labels_
# Determine which cluster corresponds to 0/1 (lower mean = 0, higher mean = 1)
cluster_means_n2 = [X_n2[labels == 0].mean(), X_n2[labels == 1].mean()]
if cluster_means_n2[0] < cluster_means_n2[1]:
    labels_n2 = 1 - labels_n2  # Swap labels if cluster order is reversed
# Calculate accuracy
accu_k_n2 = 1
for i in range(len(labels_n2)):
    if labels_n2[i] == int(d_bits[i+1]) :
        accu_k_n2 += 1
accuracy_n2 = accu_k_n2 * 100 / len(d_bits)
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(corr_bit1_n2, indices_bit1_n2, 'og', label='Bit=1')  # Green for bits=1
plt.plot(corr_bit0_n2, indices_bit0_n2, 'ob', label='Bit=0')  # Blue for bits=0
plt.axvline(x=kmeans_n2.cluster_centers_.mean(), color='r', linestyle='--', label='Cluster Boundary')
plt.legend()
plt.xlabel("Correlation Value")
plt.ylabel("Bit Position")
plt.title(f"K-means Clustering with noise (Sigma = 2) (Accuracy: {accuracy_n2}%)")
plt.savefig('results/bigmac/n2_KMEANS.png', dpi=300, bbox_inches='tight')

##################################    NOISE == 3   MEADIANE ##############################################
noised_3_s = add_noise(t_s,0,3)
noised_3_m = add_noise(t_m,0,3)
list_s_compressed_n3 = []
list_m_compressed_n3 = []
for i in range(len(t_s)):
    list_s_compressed_n3.append(compress(noised_3_s[i]))
    list_m_compressed_n3.append(compress(noised_3_m[i]))
list_corr_n3 = []
for i in range(2,len(list_s_compressed_n3)):
    list_corr_n3.append(np.corrcoef(list_s_compressed_n3[i], list_m_compressed_n3[i-1])[0][1])
med_n3 = np.median(list_corr_n3)
accu_n3 = 1
for i in range(len(list_corr_n3)):
    if list_corr_n3[i] > med_n3 and int(bin(d)[2:][i+1]) == 0:
        accu_n3 += 1
    elif list_corr_n3[i] < med_n3 and int(bin(d)[2:][i+1]) == 1:
        accu_n3 += 1
# Separate correlation values based on the bit value
corr_bit1_n3 = [list_corr_n3[i] for i in range(len(list_corr_n3)) if d_bits[i+1] == '1']
corr_bit0_n3 = [list_corr_n3[i] for i in range(len(list_corr_n3)) if d_bits[i+1] == '0']
indices_bit1_n3 = [i for i in range(len(list_corr_n3)) if d_bits[i+1] == '1']
indices_bit0_n3 = [i for i in range(len(list_corr_n3)) if d_bits[i+1] == '0']
# Plot
plt.figure(figsize=(10, 6))
plt.plot(corr_bit1_n3, indices_bit1_n3, 'og', label='Bit=1')  # Green for bits=1
plt.plot(corr_bit0_n3, indices_bit0_n3, 'ob', label='Bit=0')  # Blue for bits=0
plt.axvline(x=med_n3, color='r', label='Median')
plt.legend()
plt.xlabel("Correlation value")
plt.ylabel("Nb pairs")
plt.title(f"Collision Correlation Attack with noise (Sigma = 3) (Accuracy: {accu_n3*100/len(d_bits):.2f}%)")
plt.savefig('results/bigmac/n3_MEDIANE.png', dpi=300, bbox_inches='tight')

##################################    NOISE == 3   KMEANS ##############################################
# Reshape list_corr for K-means (requires 2D array)
X_n3 = np.array(list_corr_n3).reshape(-1, 1)
kmeans_n3 = KMeans(n_clusters=2, random_state=42).fit(X_n3)
labels_n3 = kmeans_n3.labels_
# Determine which cluster corresponds to 0/1 (lower mean = 0, higher mean = 1)
cluster_means_n3 = [X_n3[labels == 0].mean(), X_n3[labels == 1].mean()]
if cluster_means_n3[0] < cluster_means_n3[1]:
    labels_n3 = 1 - labels_n3  # Swap labels if cluster order is reversed
# Calculate accuracy
accu_k_n3 = 1
for i in range(len(labels_n3)):
    if labels_n3[i] == int(d_bits[i+1]) :
        accu_k_n3 += 1
accuracy_n3 = accu_k_n3 * 100 / len(d_bits)
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(corr_bit1_n3, indices_bit1_n3, 'og', label='Bit=1')  # Green for bits=1
plt.plot(corr_bit0_n3, indices_bit0_n3, 'ob', label='Bit=0')  # Blue for bits=0
plt.axvline(x=kmeans_n3.cluster_centers_.mean(), color='r', linestyle='--', label='Cluster Boundary')
plt.legend()
plt.xlabel("Correlation Value")
plt.ylabel("Bit Position")
plt.title(f"K-means Clustering with noise (Sigma = 3) (Accuracy: {accuracy_n3}%)")
plt.savefig('results/bigmac/n3_KMEANS.png', dpi=300, bbox_inches='tight')

##################################    NOISE == 7   MEDIANE ##############################################
noised_7_s = add_noise(t_s,0,7)
noised_7_m = add_noise(t_m,0,7)
list_s_compressed_n7 = []
list_m_compressed_n7 = []
for i in range(len(t_s)):
    list_s_compressed_n7.append(compress(noised_7_s[i]))
    list_m_compressed_n7.append(compress(noised_7_m[i]))
list_corr_n7 = []
for i in range(2,len(list_s_compressed_n7)):
    list_corr_n7.append(np.corrcoef(list_s_compressed_n7[i], list_m_compressed_n7[i-1])[0][1])

med_n7 = np.median(list_corr_n7)
accu_n7 = 1
for i in range(len(list_corr_n7)):
    if list_corr_n7[i] > med_n7 and int(bin(d)[2:][i+1]) == 0:
        accu_n7 += 1
    elif list_corr_n7[i] < med_n7 and int(bin(d)[2:][i+1]) == 1:
        accu_n7 += 1
# Separate correlation values based on the bit value
corr_bit1_n7 = [list_corr_n7[i] for i in range(len(list_corr_n7)) if d_bits[i+1] == '1']
corr_bit0_n7 = [list_corr_n7[i] for i in range(len(list_corr_n7)) if d_bits[i+1] == '0']
indices_bit1_n7 = [i for i in range(len(list_corr_n7)) if d_bits[i+1] == '1']
indices_bit0_n7 = [i for i in range(len(list_corr_n7)) if d_bits[i+1] == '0']
# Plot
plt.figure(figsize=(10, 6))
plt.plot(corr_bit1_n7, indices_bit1_n7, 'og', label='Bit=1')  # Green for bits=1
plt.plot(corr_bit0_n7, indices_bit0_n7, 'ob', label='Bit=0')  # Blue for bits=0
plt.axvline(x=med_n7, color='r', label='Median')
plt.legend()
plt.xlabel("Correlation value")
plt.ylabel("Nb pairs")
plt.title(f"Collision Correlation Attack with noise (Sigma = 7) (Accuracy: {accu_n7*100/len(d_bits):.2f}%)")
plt.savefig('results/bigmac/n7_MEDIANE.png', dpi=300, bbox_inches='tight')

##################################    NOISE == 7   KMEANS ##############################################
# Reshape list_corr for K-means (requires 2D array)
X_n7 = np.array(list_corr_n7).reshape(-1, 1)
kmeans_n7 = KMeans(n_clusters=2, random_state=42).fit(X_n7)
labels_n7 = kmeans_n7.labels_
# Determine which cluster corresponds to 0/1 (lower mean = 0, higher mean = 1)
cluster_means_n7 = [X_n7[labels == 0].mean(), X_n7[labels == 1].mean()]
if cluster_means_n7[0] < cluster_means_n7[1]:
    labels_n7 = 1 - labels_n7  # Swap labels if cluster order is reversed
# Calculate accuracy
accu_k_n7 = 1
for i in range(len(labels_n7)):
    if labels_n7[i] == int(d_bits[i+1]) :
        accu_k_n7 += 1
accuracy_n7 = accu_k_n7 * 100 / len(d_bits)
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(corr_bit1_n7, indices_bit1_n7, 'og', label='Bit=1')  # Green for bits=1
plt.plot(corr_bit0_n7, indices_bit0_n7, 'ob', label='Bit=0')  # Blue for bits=0
plt.axvline(x=kmeans_n7.cluster_centers_.mean(), color='r', linestyle='--', label='Cluster Boundary')
plt.legend()
plt.xlabel("Correlation Value")
plt.ylabel("Bit Position")
plt.title(f"K-means Clustering with noise (Sigma = 7) (Accuracy: {accuracy_n7}%)")
plt.savefig('results/bigmac/n7_KMEANS.png', dpi=300, bbox_inches='tight')

##################################    NOISE == 10   MEDIANE ##############################################
noised_10_s = add_noise(t_s,0,10)
noised_10_m = add_noise(t_m,0,10)
list_s_compressed_n10 = []
list_m_compressed_n10 = []
for i in range(len(t_s)):
    list_s_compressed_n10.append(compress(noised_10_s[i]))
    list_m_compressed_n10.append(compress(noised_10_m[i]))
list_corr_n10 = []
for i in range(2,len(list_s_compressed_n10)):
    list_corr_n10.append(np.corrcoef(list_s_compressed_n10[i], list_m_compressed_n10[i-1])[0][1])
med_n10 = np.median(list_corr_n10)
accu_n10 = 1
for i in range(len(list_corr_n10)):
    if list_corr_n10[i] > med_n10 and int(bin(d)[2:][i+1]) == 0:
        accu_n10 += 1
    elif list_corr_n10[i] < med_n10 and int(bin(d)[2:][i+1]) == 1:
        accu_n10 += 1
# Separate correlation values based on the bit value
corr_bit1_n10 = [list_corr_n10[i] for i in range(len(list_corr_n10)) if d_bits[i+1] == '1']
corr_bit0_n10 = [list_corr_n10[i] for i in range(len(list_corr_n10)) if d_bits[i+1] == '0']
indices_bit1_n10 = [i for i in range(len(list_corr_n10)) if d_bits[i+1] == '1']
indices_bit0_n10 = [i for i in range(len(list_corr_n10)) if d_bits[i+1] == '0']
# Plot
plt.figure(figsize=(10, 6))
plt.plot(corr_bit1_n10, indices_bit1_n10, 'og', label='Bit=1')  # Green for bits=1
plt.plot(corr_bit0_n10, indices_bit0_n10, 'ob', label='Bit=0')  # Blue for bits=0
plt.axvline(x=med_n10, color='r', label='Median')
plt.legend()
plt.xlabel("Correlation value")
plt.ylabel("Nb pairs")
plt.title(f"Collision Correlation Attack with noise (Sigma = 10) (Accuracy: {accu_n10*100/len(d_bits):.2f}%)")
plt.savefig('results/bigmac/n10_MEDIANE.png', dpi=300, bbox_inches='tight')

##################################    NOISE == 10   KMEANS ##############################################
# Reshape list_corr for K-means (requires 2D array)
X_n10 = np.array(list_corr_n10).reshape(-1, 1)
kmeans_n10 = KMeans(n_clusters=2, random_state=42).fit(X_n10)
labels_n10 = kmeans_n10.labels_
# Determine which cluster corresponds to 0/1 (lower mean = 0, higher mean = 1)
cluster_means_n10 = [X_n10[labels_n10 == 0].mean(), X_n10[labels_n10 == 1].mean()]
if cluster_means_n10[0] < cluster_means_n10[1]:
    labels_n10 = 1 - labels_n10  # Swap labels if cluster order is reversed
# Calculate accuracy
accu_k_n10 = 1
for i in range(len(labels_n10)):
    if labels_n10[i] == int(d_bits[i+1]):
        accu_k_n10 += 1
accuracy_n10 = accu_k_n10 * 100 / len(d_bits)
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(corr_bit1_n10, indices_bit1_n10, 'og', label='Bit=1')  # Green for bits=1
plt.plot(corr_bit0_n10, indices_bit0_n10, 'ob', label='Bit=0')  # Blue for bits=0
plt.axvline(x=kmeans_n10.cluster_centers_.mean(), color='r', linestyle='--', label='Cluster Boundary')
plt.legend()
plt.xlabel("Correlation Value")
plt.ylabel("Bit Position")
plt.title(f"K-means Clustering with noise (Sigma = 10) (Accuracy: {accuracy_n10}%)")
plt.savefig('results/bigmac/n10_KMEANS.png', dpi=300, bbox_inches='tight')

##################################    RESUME   MEDIANE ##############################################
plt.figure(figsize=(10, 6))
plt.plot([0,1,2,3,7,10], [accu*100/len(d_bits), accu_n1*100/len(d_bits), accu_n2*100/len(d_bits), accu_n3*100/len(d_bits), accu_n7*100/len(d_bits), accu_n10*100/len(d_bits)])
plt.plot([0,1,2,3,7,10],[accu*100/len(d_bits), accu_n1*100/len(d_bits), accu_n2*100/len(d_bits), accu_n3*100/len(d_bits), accu_n7*100/len(d_bits), accu_n10*100/len(d_bits)], 'or')
plt.yticks([accu*100/len(d_bits), accu_n1*100/len(d_bits), accu_n2*100/len(d_bits), accu_n3*100/len(d_bits), accu_n7*100/len(d_bits), accu_n10*100/len(d_bits)])
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.ylabel("Accuracy in %")
plt.xlabel("Level of Noise (Sigma Value)")
plt.title("Impact of Noise on Collision Correlation Attack with BigMac Pre-Processing with Median Threshold")
plt.savefig('results/resume_MEDIANE.png', dpi=300, bbox_inches='tight')

##################################    RESUME  KMEANS ##############################################
plt.figure(figsize=(10, 6))
plt.plot([0,1,2,3,7,10], [accuracy, accuracy_n1, accuracy_n2, accuracy_n3, accuracy_n7, accuracy_n10])
plt.plot([0,1,2,3,7,10], [accuracy, accuracy_n1, accuracy_n2, accuracy_n3, accuracy_n7, accuracy_n10], 'or')
plt.yticks([accuracy, accuracy_n1, accuracy_n2, accuracy_n3, accuracy_n7, accuracy_n10])
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.ylabel("Accuracy in %")
plt.xlabel("Level of Noise (Sigma Value)")
plt.title("Impact of Noise on Collision Correlation Attack with BigMac Pre-Processing with K-means threshold")
plt.savefig('results/resume_KMEANS.png', dpi=300, bbox_inches='tight')