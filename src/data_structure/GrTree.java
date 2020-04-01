package data_structure;


import constants.PredicateConstants;
import helper.Converter;
import helper.Sorter;
import helper.SupportCounter;

import java.util.*;

public class GrTree {
    public ArrayList<Predicate> headTable;
    public HashSet<Integer> prefix;
    public GrNode root;

    public class GrNode {
        public HashMap<Integer, GrNode> children;
        public Predicate predicate;

        public GrNode(Integer pId) {
            children = new HashMap<>();
            predicate = new Predicate(pId, 0, 0);
        }

        public GrNode(Predicate p) {
            children = new HashMap<>();
            predicate = new Predicate(p.id, p.plusSupport, p.negativeSupport);
        }
    }

    public GrTree(ArrayList<ArrayList<Predicate>> database, ArrayList<Support> dataClasses) {
        HashSet<Integer> prefix = new HashSet<>();
        createGrTree(database, dataClasses, prefix);
    }

    public GrTree(ArrayList<ArrayList<Predicate>> database, ArrayList<Support> dataClasses, HashSet<Integer> prefix) {
        createGrTree(database, dataClasses, prefix);
    }

    public GrTree(ConditionalDatabase conditionalDatabase) {
        createGrTree(conditionalDatabase.database, conditionalDatabase.dataClasses, conditionalDatabase.prefix);
    }


    private void removeAllPredicateWithFullSupport() {
        HashSet<Integer> predicateIdsToRemove = new HashSet<>();

        while (root.children.size() == 1) {


            Integer childId = root.children.keySet().iterator().next();
            predicateIdsToRemove.add(childId);
            GrNode child = root.children.get(childId);

            root.children.remove(childId);

            if (child.children.size() >= 1) {
                for (Integer grandChildrenId: child.children.keySet()) {
                    GrNode grandChildren = child.children.get(grandChildrenId);
                    root.children.put(grandChildrenId, grandChildren);
                }
            }
        }

        removePredicatesFromHeadTable(predicateIdsToRemove);
    }

    private void removePredicatesFromHeadTable(HashSet<Integer> predicateIds) {
        ArrayList<Predicate> newHeadTable = new ArrayList<>();

        for (Predicate p: headTable) {
            if (!predicateIds.contains(p.id)) {
                newHeadTable.add(p);
            }
        }

        headTable = newHeadTable;
    }

    private void createGrTree(ArrayList<ArrayList<Predicate>> database, ArrayList<Support> dataClasses, HashSet<Integer> prefix) {
        // construct headTable and dataClasses
        ArrayList<Predicate> arrPredicate = SupportCounter.getSupportAllPredicate(database, dataClasses);
        this.headTable = Sorter.sortArrayBySupport(arrPredicate);

        // construct sorted database
        HashMap<Integer, Support> counter = Converter.convertPredicateArrayToHashMap(arrPredicate);
        Converter.fillDatabaseWithSupport(database, counter);
        ArrayList<ArrayList<Predicate>> sortedDatabase= Sorter.generateSortedDatabase(database);

        // construct Trie
        root = new GrNode(new Predicate(-1, -1, -1));
        for (int i = 0; i < sortedDatabase.size(); i++) {
            insertToTrie(sortedDatabase.get(i), dataClasses.get(i));
        }

        // construct prefix
        this.prefix = new HashSet<>();
        this.prefix.addAll(prefix);

        // delete predicate with full support
        removeAllPredicateWithFullSupport();
    }

    public void insertToTrie(ArrayList<Predicate> arr, Support dataClass) {
        GrNode currentNode = root;
        for (Predicate p: arr) {
            if (!currentNode.children.containsKey(p.id)) {
                currentNode.children.put(p.id, new GrNode(p.id));
            }
            currentNode = currentNode.children.get(p.id);
            currentNode.predicate.plusSupport += dataClass.plusSupport;
            currentNode.predicate.negativeSupport += dataClass.negativeSupport;
        }
    }

    public void printHeadTable() {
        if (root == null) {
            return;
        }
        System.out.println(Arrays.toString(headTable.toArray()));
    }


    public Support countUnavoidableSupport() {
      if (root.children.size() == 0) {
          return new Support(0, 0);
      }

      Support result = new Support(-1, -1);
      for (GrNode child: root.children.values()) {
         countUnavoidableSupportRecurs(child, 1, result);
      }

      if (result.plusSupport != -1  && result.negativeSupport != -1) {
          return result;
      }

      return countUnavoidableSupportFromLeaf();
    }

    private void countUnavoidableSupportRecurs(GrNode node, Integer pathSize, Support result) {
        if (pathSize == this.headTable.size()) {
            result.plusSupport = node.predicate.plusSupport;
            result.negativeSupport = node.predicate.negativeSupport;
            return;
        }

        for (GrNode child: node.children.values()) {
            countUnavoidableSupportRecurs(child, pathSize+1,result);
        }
    }

    private ArrayList<GrNode> getAllLeafNodes() {
        ArrayList<GrNode> result = new ArrayList<>();
        getAllLeafNodesRecurs(root, result);
        return result;
    }

    private void getAllLeafNodesRecurs(GrNode node, ArrayList<GrNode> result) {
        if (node.children.size() == 0) {
            result.add(node);
            return;
        }

        for (GrNode child: node.children.values()) {
            getAllLeafNodesRecurs(child, result);
        }
    }

    private Support countUnavoidableSupportFromLeaf() {
        ArrayList<GrNode> leafNodes = getAllLeafNodes();

        Integer minPlusSupport = Integer.MAX_VALUE;
        Integer minNegativeSupport = Integer.MAX_VALUE;

        for (GrNode node: leafNodes) {
            Integer nodePlusSupport = node.predicate.plusSupport;
            Integer nodeNegativeSupport = node.predicate.negativeSupport;

            if (nodePlusSupport < minPlusSupport) {
                minPlusSupport = nodePlusSupport;
            }

            if (nodeNegativeSupport < minNegativeSupport) {
                minNegativeSupport = nodeNegativeSupport;
            }
        }

        return new Support(minPlusSupport, minNegativeSupport);
    }

    public void printTrie() {
        for (Map.Entry<Integer, GrNode> entry: this.root.children.entrySet()) {
            printRecurs(entry.getValue(), new ArrayList<Predicate>());
        }
    }

    private void printRecurs(GrNode gn, ArrayList<Predicate> path) {
        if (gn.children.size() == 0) {
            path.add(gn.predicate);
            System.out.println(Arrays.toString(path.toArray()));
            path.remove(path.size()-1);

            return;
        }
        path.add(gn.predicate);
        for (Map.Entry<Integer, GrNode> entry: gn.children.entrySet()) {
            printRecurs(entry.getValue(), path);
        }
        path.remove(path.size()-1);
    }


}
