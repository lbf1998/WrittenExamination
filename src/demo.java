import utils.TreeNode;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class demo {


    public static void main(String[] args) {
        int[] nodes = {1,2,3,4,5,6,7,8,8,9};
        TreeNode root = getTree(nodes, 0);
        System.out.println(hasPathSum(root, 10));
    }

    private static TreeNode getTree(int[] nodes, int index) {
        TreeNode tn = null;
        if (index < nodes.length){
            Integer value  = nodes[index];
            if (value != null){
                return null;
            }
            tn = new TreeNode(value);
            tn.left = getTree(nodes, 2*index+1);
            tn.right = getTree(nodes, 2*index+2);
            return tn;
        }
        return tn;
    }

    static List<TreeNode> list;

    public static boolean hasPathSum(TreeNode root, int targetSum) {

        return dfs(root, targetSum, 0, list);
    }
    public static boolean dfs(TreeNode root, int targetSum, int now_sum, List<TreeNode> list){
        return false;
    }
}

