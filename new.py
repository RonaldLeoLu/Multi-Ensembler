from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from tree import RandomDecisionTreeClassifier
from Randomlearners import MajorVoting
from sklearn.neighbors import KNeighborsClassifier
#from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from Ensembler import Bootstrap,Softmax
weight = np.repeat(1/x_tr.shape[0],x_tr.shape[0])
def proc_w(l1,l2,w):
    error = accuracy_score(l1,l2)
    
    alpha = 1/2 * np.log((1-error) / max(error, 1e-6))
    
    c_idx = []
    
    for idx,v in enumerate(l1):
        if v != l2[idx]:
            c_idx.append(idx)
            
    for i in range(len(w)):
        if i in c_idx:
            w[i] = w[i]*np.exp(alpha)
        else:
            w[i] = w[i]*np.exp(-1*alpha)
            
    zt = np.sum(w)
    
    w = np.array(w) / zt
    return w


clf1 = LogisticRegression(solver='sag')
clf2 = lda()
clf3 = DecisionTreeClassifier(max_depth=2)
clf4 = KNeighborsClassifier(n_neighbors=1)
clf5 = KNeighborsClassifier(n_neighbors=10)
clfs = [clf1,clf2,clf3,clf4,clf5]

layer2_tr = None
layer2_te = None
for idx,model in enumerate(clfs):
    for i in range(5):
        x,y = Bootstrap(x_tr,y_tr,weight=weight,factor = 0.8)
        model.fit(x, y)
        prdtr = model.predict(x_tr)
        prdte = model.predict(x_te)
        
        if layer2_tr is None:
            layer2_tr = np.array(prdtr)
        else:
            layer2_tr = np.column_stack([layer2_tr,prdtr])
            
        if layer2_te is None:
            layer2_te = np.array(prdte)
        else:
            layer2_te = np.column_stack([layer2_te,prdte])
            
layer3_tr = None
layer3_te = None

for model in [clf1,clf2,clf5]:
    model.fit(layer2_tr, y_tr)
    prdtr = model.predict(layer2_tr)
    prdte = model.predict(layer2_te)
    
    if layer3_tr is None:
        layer3_tr = np.array(prdtr)
    else:
        layer3_tr = np.column_stack([layer3_tr,prdtr])
            
    if layer3_te is None:
        layer3_te = np.array(prdte)
    else:
        layer3_te = np.column_stack([layer3_te,prdte])
    
m = MajorVoting()
trprd = m.fit_transform(layer3_tr)
teprd = m.fit_transform(layer3_te)

print('Train score: {:.6f}'.format(accuracy_score(trprd,y_tr)))
print('Test score: {:.6f}'.format(accuracy_score(teprd,y_te)))


for clf in [clf1,clf2,clf3,clf4,clf5]:
    clf.fit(x_tr,y_tr)
    prd = clf.predict(x_tr)
    print(accuracy_score(prd,y_tr))