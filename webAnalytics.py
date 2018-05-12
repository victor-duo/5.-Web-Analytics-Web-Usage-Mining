import pandas as pd
from collections import Counter
import networkx as nx

def frequentItems(transactions, support):
    counter = Counter()
    for trans in transactions:
        counter.update(frozenset([t]) for t in trans)
    return set(item for item in counter if
               float(counter[item])/len(transactions) >= support), counter


def generateCandidates(L, k):
    candidates = set()
    for a in L:
        for b in L:
            union = a | b
            if len(union) == k and a != b:
                candidates.add(union)
    return candidates


def filterCandidates(transactions, itemsets, support):
    counter = Counter()
    for trans in transactions:
        subsets = [itemset for itemset in itemsets if itemset.issubset(trans)]
        counter.update(subsets)
    return set(item for item in counter if
               float(counter[item])/len(transactions) >= support), counter


def apriori(transactions, support):
    result = list()
    resultc = Counter()
    candidates, counter = frequentItems(transactions, support)
    result += candidates
    resultc += counter
    k = 2
    while candidates:
        candidates = generateCandidates(candidates, k)
        candidates, counter = filterCandidates(transactions,
                                               candidates, support)
        result += candidates
        resultc += counter
        k += 1
    resultc = {item: (float(resultc[item])/len(transactions)) for item in
               resultc}
    return result, resultc


def genereateRules(frequentItemsets, supports, minConfidence, sorting):
    array = []
    for f in frequentItemsets:
        if(len(f) > 1):
            for sub in f:
                C = frozenset([sub])
                IminusC = frozenset(f)-C
                confidence = supports[f]/float(supports[IminusC])
                lift = float(supports[f])/(supports[IminusC]*supports[C])
                if(confidence >= minConfidence):
                    array.append([IminusC, C, supports[f], confidence, lift])
                    # print("{} - {}, support = {}, confidence = {}, lift = {}"
                    #      .format(IminusC, C, supports[C], confidence, lift))
    array = sorted(array, key=lambda x: x[sorting])
    for row in array:
        IminusC = []
        for string in row[0]:
            IminusC.append(string)
        C = []
        for string in row[1]:
            C.append(string)
        print("{} - {}, support = {}, confidence = {}, lift = {}.".
              format(IminusC, C, row[2], row[3], row[4]))

df2 = pd.read_csv("search_engine_map.csv")
referrer = {}
listReferrer = {}
validReferrer = []
for index,row in df2.iterrows():
    keys = df2.keys()
    if(str(row[keys[1]]) != 'nan'):
        # listReferrer[str(row[keys[0]])] = str(row[keys[1]])
        validReferrer.append(str(row[keys[0]]))
# for key in listReferrer:
#     if(listReferrer[key] not in referrer):
#         referrer[listReferrer[key]] = []
#     if(key not in referrer[listReferrer[key]]):
#         referrer[listReferrer[key]].append(key)
# referrer is the dict where for one type of referrer domain, you get all
# the anonymized referrer

df1 = pd.read_csv("visitors.csv")
datasetVisitor = []
validVisitor = []
print(len(df1))
for index,row in df1.iterrows():
    if(row[1] not in validReferrer): continue
    if(row[4] == 0): continue
    validVisitor.append(row[0])
    row = [col+"="+str(row[col]) for col in list(df1)]
    datasetVisitor.append(row)
print(len(datasetVisitor))
dataset = []
df = pd.read_csv("clicks.csv")
del df["CatID"]
del df["ExtCatID"]
del df["ExtCatName"]
del df["LocalID"]
del df["PageID"]
del df["TopicID"]
del df["TopicName"]
nbApplications = 0
nbCatalogs = 0
nbDiscounts = 0
nbHowToJoin = 0
nbInsurances = 0
nbWhoWeAre = 0
for index,row in df.iterrows():
    if(row[0] not in validVisitor): continue
    if(row[1] == "APPLICATION"): nbApplications += 1
    elif(row[1] == "CATALOG"): nbCatalogs += 1
    elif(row[1] == "DISCOUNT"): nbDiscounts += 1
    elif(row[1] == "HOWTOJOIN"): nbHowToJoin += 1
    elif(row[1] == "INSURANCE"): nbInsurances += 1
    elif(row[1] == "WHOWEARE"): nbWhoWeAre += 1
    formatedRow = []
    for col in list(df):
        formatedRow.append(col+"="+str(row[col]))
    dataset.append(formatedRow)
print(len(dataset))
print("Applications: {}, Catalogs: {}, Discounts: {}, HowToJoin: {}, ".
format(nbApplications,nbCatalogs,nbDiscounts,nbHowToJoin) +
"Insurances: {}, WhoWeAre: {}".format(nbInsurances,nbWhoWeAre))

frequentItemsets, supports = apriori(dataset, 0.1)
genereateRules(frequentItemsets, supports, 0.5, 2)

print("-----------------------------------")
frequentItemsets, supports = apriori(datasetVisitor, 0.02)
genereateRules(frequentItemsets, supports, 0.5, 2)


G = nx.Graph()
visitedNode = []
for row1 in dataset:
    if(row1[0] not in visitedNode):
        G.add_node(row1[0], label=row1[0])
        visitedNode.append(row1[0])
for row1 in dataset:
    for row2 in dataset:
        if(row1[0] != row2[0] and row1[1] == row2[1]):
            # They are linked because they went to the same Page
            G.add_edge(row1[0], row2[0], label=row1[1], weight=1)
print("Number of nodes: %s" % nx.number_of_nodes(G))
print("Number of edges: %s" % nx.number_of_edges(G))
print("Density: %s" % nx.density(G))
centralTab = {}
centralities = [nx.degree_centrality, nx.eigenvector_centrality]
for centrality in centralities:
    topNode = []
    print(centrality.__name__)
    result = centrality(G)
    for k in result:
        valueWithIndex = [result[k], k]
        topNode = sorted(topNode, key=lambda value: value[0],
                         reverse=True)[:10]
        topNode.append(valueWithIndex)
    centralTab[centrality.__name__] = topNode
print(centralTab)
#If you want to see data from the top visitor which has ID = 12350
#for row in dataset:
#    if(row[0]=="VisitID=12350"): print(row)
