import numpy as np

nro_clases = 20
#nro_clases = 6

def pesosTFIDF(sparse_matrix_tfidf,df_train):
  col = sparse_matrix_tfidf.shape[1]
  f = np.zeros((nro_clases,col))
  for i in range(0,nro_clases):
    target_bool = np.array(df_train['clase'] == i)
    r = (sparse_matrix_tfidf[target_bool,:].sum(axis=0))
    r = r.getA1().tolist()
    f[i,:]=np.array(r)
  return f

#Probailidades condicionales
def probaxclase(sparse_matrix,df_train,alpha):
  #input: matriz sparse salida del count vectorizer
  #output: matriz(j,n) con probabilidad P(xn|yj) siendo n cada atributo y j cada clase
  col = sparse_matrix.shape[1]
  f = np.zeros((nro_clases,col))
  for i in range(0,nro_clases):
    target_bool = np.array(df_train['clase'] == i)
    r = (sparse_matrix[target_bool,:].sum(axis=0)+alpha)/(sparse_matrix[target_bool,:].sum()+alpha*col)
    r = r.getA1().tolist()
    f[i,:]=np.array(r)
  return f

#probailidades a priori
def probaPriori(df_train):
  #output: fila con la probabilidad a priori de cada clase en orden 
  r=[]
  for i in range(0,nro_clases):
    target_bool = np.array(df_train['clase'] == i)
    target = 1*target_bool
    r.append(np.sum(target)/len(target))
  return r


def likelihood(probas_cond,sparse_matrix):
  filas = sparse_matrix.shape[0]
  r = []
  f = np.zeros((filas,nro_clases))
  for c in range(0,nro_clases):
    for i in range(0,filas):
      #el likelihood del texto i de pertenecer a la clase c
      l = np.prod(probas_cond[c,:]**np.array(sparse_matrix[i].todense()[0]))
      r.append(l)
    
    f[:,c] = np.array(r)
    r=[]
  return 

def log_likelihood(probas_cond,sparse_matrix):
  filas = sparse_matrix.shape[0]
  r = []
  f = np.zeros((filas,nro_clases))
  for c in range(0,nro_clases):
    p = np.log(probas_cond[c,:])
    for i in range(0,filas):
      l = np.dot(np.array(p),np.array(sparse_matrix[i].todense()[0])[0])
      r.append(l)
    f[:,c] = np.array(r)
    r=[]
  return f

def log_likelihood2(probas_cond,sparse_matrix,weights):
  filas = sparse_matrix.shape[0]
  r = []
  f = np.zeros((filas,nro_clases))
  for c in range(0,nro_clases):
    p = np.log(probas_cond[c,:]*weights[c,:]+1e-30)
    for i in range(0,filas):
      l = np.dot(np.array(p),np.array(sparse_matrix[i].todense()[0])[0])
      r.append(l)
    f[:,c] = np.array(r)
    r=[]
  return f

def probaPost(probas_cond,priori,sparse_matrix,log_like):
  r=[]
  filas = sparse_matrix.shape[0]
  final= np.zeros((filas,nro_clases))
  if log_like:
    l = log_likelihood(probas_cond,sparse_matrix)
  else:
    l = likelihood(probas_cond,sparse_matrix)
  for c in range(0,nro_clases):
    if log_like:
      l_c = np.array(l[:,c])+np.log(priori[c])
    else:
      l_c = np.array(l[:,c])*priori[c]
    final[:,c] = np.array(l_c)
  return final

def probaPost2(probas_cond,priori,sparse_matrix,log_like,weights):
  r=[]
  filas = sparse_matrix.shape[0]
  final= np.zeros((filas,nro_clases))
  if log_like:
    l = log_likelihood2(probas_cond,sparse_matrix,weights)
  else:
    l = likelihood(probas_cond,sparse_matrix)
  for c in range(0,nro_clases):
    if log_like:
      l_c = np.array(l[:,c])+np.log(priori[c])
    else:
      l_c = np.array(l[:,c])*priori[c]
    final[:,c] = np.array(l_c)
  return final

def clasificacion(proba_posteriori):
  filas = proba_posteriori.shape[0] 
  clase = np.zeros((filas,1))
  for i in range(0,filas):
    post = proba_posteriori[i,:]
    c = np.where(post == np.max(post))[0][0]
    clase[i] = c
  return clase
