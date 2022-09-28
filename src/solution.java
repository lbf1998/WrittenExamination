
import sun.awt.image.ImageWatched;
import utils.ListNode;
import utils.TreeNode;

import java.util.*;

public class solution {
    public static void main(String[] args) {
        num1 num1 = new num1();

        String s = "aabbaac";
        String commonStr = num1.longestPalindrome1(s);

//        String len = n.longestPalindrome("abcdehasjdd  jkel");
        System.out.println(commonStr);
    }
}

/*
    1.两数之和
   给定一个整数数组 nums 和一个整数目标值 target，
   请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
采用hashmap,一次遍历
 */
class a1_twoSum{

    public static int[] twoSum1(int[] nums, int target) {
        Map<Integer, Integer> hashtable = new HashMap<Integer, Integer>();
        for (int i = 0; i < nums.length; i++) {
            if (hashtable.containsKey(target - nums[i])) {
                return new int[]{hashtable.get(target - nums[i]), i};
            }
            hashtable.put(nums[i], i);
        }
        return new int[0];
    }
}

/*
    2.链表相加
    返回头结点指针。
    注意不要忘记余数。
*/
class a2_addTwoNumbers{
    public static ListNode addTwoNumbers1(ListNode l1, ListNode l2) {
        ListNode l_sum = new ListNode(0);
        ListNode head = l_sum;
        int carry = 0;
        while (l1 != null || l2 != null) {
            int val1 = l1 == null ? 0 : l1.val;
            int val2 = l2 == null ? 0 : l2.val;
            int sum = val1 + val2 + carry;
            carry = sum / 10;
            sum = sum % 10;
            l_sum.next = new ListNode(sum);
            l_sum = l_sum.next;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (carry != 0) {
            l_sum.next = new ListNode(carry);
        }
        return head;
    }

}

class num1 {



    /*
       3.返回不重复最长子串的长度
    利用hashmap存储所有字母开头的字符串的长度。

    本题利用滑动窗口来进行求解，每次遇到重复的字符串，移动到上次遇到字符串的最后位置，即：将滑动窗口收缩。
    只要没有在map中出现过的字符一定是不会重复的字符，出现过就说明，目前已经重复了。
    Map存储的是每个字符串位置加1

     */
    public int lengthOfLongestSubstring(String s) {
        int len = s.length();
        int max = 0;
        int left = -1;
        Set<Character> occ = new HashSet<>();
        for (int i = 0; i < len; i++) {
            if (i != 0) {
                occ.remove(s.charAt(i));
            }
            while (left + 1 < len && !occ.contains(s.charAt(i))) {
                occ.add(s.charAt(i));
                left++;
            }
            max = Math.max(max, left + i - 1);
        }
        return max;
    }

    /*
    5.找出最长回文子串

    1.利用动态规划（利用表格，当前字符是否是回文与表格左下角的值有关，如果出现表格右上角不是回文的情况，那么这个字符就不是回文。
    2.中心扩展法（与暴力枚举法相反）
     */
    public String longestPalindrome(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }
        int strLen = s.length();
        int left = 0;
        int right = 0;
        int len = 1;
        int maxStart = 0;
        int maxLen = 0;

        for (int i = 0; i < strLen; i++) {
            left = i - 1;
            right = i + 1;
            while (left >= 0 && s.charAt(left) == s.charAt(i)) {
                len++;
                left--;
            }
            while (right < strLen && s.charAt(right) == s.charAt(i)) {
                len++;
                right++;
            }
            while (left >= 0 && right < strLen && s.charAt(right) == s.charAt(left)) {
                len = len + 2;
                left--;
                right++;
            }
            if (len > maxLen) {
                maxLen = len;
                maxStart = left;
            }
            len = 1;
        }
        return s.substring(maxStart + 1, maxStart + maxLen + 1);
    }

    // 动态规划法：利用表格进行判断，每次计算表格右上角的值
    public String longestPalindrome2(String s) {
        int len = s.length();
        if (len < 2) {
            return s;
        }
        int maxLen = 1;
        int begin = 0;
        // dp[i][j] 表示 s[i..j] 是否是回文串
        boolean[][] dp = new boolean[len][len];
        // 初始化：所有长度为 1 的子串都是回文串
        for (int i = 0; i < len; i++) {
            dp[i][i] = true;
        }

        char[] charArray = s.toCharArray();
        // 递推开始
        // 先枚举子串长度
        for (int L = 2; L <= len; L++) {
            // 枚举左边界，左边界的上限设置可以宽松一些
            for (int i = 0; i < len; i++) {
                // 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
                int j = L + i - 1;
                // 如果右边界越界，就可以退出当前循环
                if (j >= len) {
                    break;
                }

                if (charArray[i] != charArray[j]) {
                    dp[i][j] = false;
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        // 如果此时字符相同，与表格左下角的值进行匹配
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }

                // 只要 dp[i][L] == true 成立，就表示子串 s[i..L] 是回文，此时记录回文长度和起始位置
                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substring(begin, begin + maxLen);
    }

    public String longestPalindrome1(String s) {
        int len = s.length();
        if (len == 0) return s;
        String res = "";
        int maxLen = 0;
        boolean[][] dp = new boolean[len][len];
        // 枚举每个 i---j， i代表左端点，j代表右端点。
        for (int i = 0; i < len; i++) {
            for (int j = 0; j <= i; j++) {
                // 如果i, j 相同，true
                // 如果相邻 ，字符相同则true
                if (s.charAt(i) == s.charAt(j)) {
                    if (i == j || j == i - 1) {
                        dp[j][i] = true;
                    } else if (dp[j + 1][i - 1]) {
                        dp[j][i] = true;
                    }
                }
                if (dp[j][i] && i - j + 1 > maxLen) {
                    maxLen = i - j + 1;
                    res = s.substring(j, i + 1);
                }
            }
        }
        return res;
    }

    /*
        6.Z字变换
        将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。
        思路：利用二维数组进行排列

     */
    public String convert(String s, int numRows) {
        int n = s.length(), r = numRows;
        if (r == 1 || r >= n) {
            return s;
        }
        int t = r * 2 - 2;
        int c = (n + t - 1) / t * (r - 1);
        char[][] mat = new char[r][c];
        for (int i = 0, x = 0, y = 0; i < n; ++i) {
            mat[x][y] = s.charAt(i);
            if (i % t < r - 1) {
                ++x; // 向下移动
            } else {
                --x;
                ++y; // 向右上移动
            }
        }
        StringBuffer ans = new StringBuffer();
        for (char[] row : mat) {
            for (char ch : row) {
                if (ch != 0) {
                    ans.append(ch);
                }
            }
        }
        return ans.toString();
    }

    /*
        7.整数反转
        给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。

        如果反转后整数超过 32 位的有符号整数的范围 [−231,  231 − 1] ，就返回 0。
        假设环境不允许存储 64 位整数（有符号或无符号）。
     */
    public int reverse(int x) {
        int res = 0;
        while (x != 0) {
            int tmp = res * 10 + x % 10;
            if (tmp / 10 != res) {  // 溢出
                return 0;
            }
            res = tmp;
            x /= 10;
        }
        return res;
    }

    /*
        8.public int myAtoi(String s)
        从字符中提取数字。
     */
    public int myAtoi(String s) {
        char[] chars = s.toCharArray();
        int len = chars.length;
        //1.去空格
        int index = 0;
        while (index < len && chars[index] == ' ')
            index++;
        //2.排除极端情况 "    "
        if (index == len) return 0;
        //3.设置符号
        int sign = 1;
        char firstChar = chars[index];
        if (firstChar == '-') {
            index++;
            sign = -1;
        } else if (firstChar == '+') {
            index++;
        }
        int res = 0, last = 0; //last 记录上一次的res，以此来判断是否溢出
        while (index < len) {
            char c = chars[index];
            if (c < '0' || c > '9') break;
            int tem = c - '0';
            last = res;
            res = res * 10 + tem;
            if (last != res / 10)  ////如果不相等就是溢出了
                return (sign == (-1)) ? Integer.MIN_VALUE : Integer.MAX_VALUE;
            index++;
        }
        return res * sign;
    }

    /*
    9.回文数
    给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。

    回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。


     */
    public boolean isPalindrome(int x) {
        if (x == 0) return true;
        if (x < 0 || x % 10 == 0) return false;
        int reversed = 0;
        while (x > reversed) {
            reversed = reversed * 10 + x % 10;
            x /= 10;
        }
        return x == reversed || x == reversed / 10;
    }

    /***
     *最长公共前缀
     * @param strs
     * @return
     */
    public String longestCommonPrefix(String[] strs) {

        if (strs == null || strs.length == 0)
            return "";
        String prefix = strs[0];
        int count = strs.length;
        for (int i = 0; i < count; i++) {
            prefix = longestCommonPrefix(prefix, strs[i]);
        }
        return prefix;
    }

    private String longestCommonPrefix(String prefix, String str) {
        int length = Math.min(prefix.length(), str.length());
        int index = 0;
        while (index < length && prefix.charAt(index) == str.charAt(index)) {
            index++;
        }
        return prefix.substring(0, index);

    }

    public String longestCommonPrefix2(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }
        int length = strs[0].length();
        int count = strs.length;
        for (int i = 0; i < length; i++) {
            char c = strs[0].charAt(i);
            for (int j = 1; j < count; j++) {
                if (i == strs[j].length() || strs[j].charAt(i) != c) {
                    return strs[0].substring(0, i);
                }
            }
        }
        return strs[0];
    }

    /*
    11.承最多水的容器
     给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
    思路：利用双指针，分析得出每次移动最小的那个指针肯定会找到最大值。
     */
    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1;
        int ans = 0;
        while (left < right) {
            int tmp = Math.min(height[left], height[right]) * (right - left);
            ans = Math.max(tmp, ans);
            if (height[left] > height[right]) {
                right--;
            } else {
                left++;
            }
        }
        return ans;
    }

    /*
        12.整数转换罗马数字
        利用字符匹配。
     */
    public String intToRoman(int num) {
        int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] symbols = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        StringBuffer roman = new StringBuffer();
        for (int i = 0; i < values.length; ++i) {
            int value = values[i];
            String symbol = symbols[i];
            while (num >= value) {
                num -= value;
                roman.append(symbol);
            }
            if (num == 0) {
                break;
            }
        }
        return roman.toString();
    }

    /*
    13. 罗马转数字
    利用hashmap
     */
    public int romanToInt(String s) {
        Map<Character, Integer> symbolValues = new HashMap<Character, Integer>() {{
            put('I', 1);
            put('V', 5);
            put('X', 10);
            put('L', 50);
            put('C', 100);
            put('D', 500);
            put('M', 1000);
        }};

        int ans = 0;
        int n = s.length();
        for (int i = 0; i < n; ++i) {
            int value = symbolValues.get(s.charAt(i));
            if (i < n - 1 && value < symbolValues.get(s.charAt(i + 1))) {
                ans -= value;
            } else {
                ans += value;
            }
        }
        return ans;
    }

    /**
     * 二叉树的右侧视图
     */
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        if (root == null) {
            return list;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
                if (i == size - 1) {
                    list.add(node.val);
                }
            }
        }
        return list;
    }

    public boolean isSymmetric(TreeNode root) {
        return check(root, root);
    }

    public boolean check(TreeNode u, TreeNode v) {
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(u);
        q.offer(v);
        while (!q.isEmpty()) {
            u = q.poll();
            v = q.poll();
            if (u == null && v == null) {
                continue;
            }
            if ((u == null || v == null) || (u.val != v.val)) {
                return false;
            }

            q.offer(u.left);
            q.offer(v.right);

            q.offer(u.right);
            q.offer(v.left);
        }
        return true;
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if (root==null){
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()){
            int size = queue.size();
            List<Integer> row = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                row.add(node.val);
                if (node.left!=null){
                    queue.offer(node.left);
                }
                if (node.right!=null){
                    queue.offer(node.right);
                }
            }
            res.add(row);
        }
        return res;
    }


    /***
     * LRU缓存淘汰算法
     * 最近最少未使用。
     *
     */

    class LRUCache {
        class LinkedNode{
            int key;
            int value;
            LinkedNode pre;
            LinkedNode next;
            public LinkedNode(){};
            public LinkedNode(int _key, int _value){
                this.key = _key;
                this.value = _value;
            };
        }
        private HashMap<Integer, LinkedNode> cache = new HashMap<Integer, LinkedNode>();
        private LinkedNode head;
        private LinkedNode tail;
        private int size;
        private int capacity;

        public LRUCache(int capacity) {
            this.size = 0;
            this.capacity = capacity;
            // 使用伪头部和伪尾部节点
            head = new LinkedNode();
            tail = new LinkedNode();
            head.next = tail;
            tail.pre = head;

        }

        public int get(int key) {
            LinkedNode node = cache.get(key);
            if (node==null){
                return -1;
            }
            moveNode(node);
            return node.value;
        }

        public void put(int key, int value) {
            LinkedNode node = cache.get(key);
            if (node==null){
                LinkedNode newNode = new LinkedNode(key, value);
                cache.put(key, newNode);
                addToHead(newNode);
                ++size;
                if (size > capacity){
                    LinkedNode tailNode = removeTail();
                    cache.remove(tailNode.key);
                    --size;
                }
            }else {
                node.value = value;
                moveNode(node);
            }
        }



        private void moveNode(LinkedNode node) {
            removeNode(node);
            addToHead(node);
        }

        private void addToHead(LinkedNode node) {
            node.pre = head;
            node.next = head.next;
            head.next.pre = node;
            head.next = node;
        }

        private void removeNode(LinkedNode node) {
            // 双向链表删除节点
            node.pre.next = node.next;
            node.next.pre = node.pre;
        }
        private LinkedNode removeTail() {
            LinkedNode res = tail.pre;
            removeNode(res);
            return res;
        }
    }

    //二叉树的中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        inorder(root, res);
        return res;

    }

    private void inorder(TreeNode root, List<Integer> res) {
        if (root == null) {
            return;
        }
        inorder(root.left, res);
        res.add(root.val);
        inorder(root.right, res);
    }

    private void inorderNoTraversal(TreeNode root, List<Integer> res) {
        Deque<TreeNode> stk = new LinkedList<TreeNode>();
        while (root != null || !stk.isEmpty()) {
            while (root != null) {
                stk.push(root);
                root = root.left;
            }
            root = stk.pop();
            res.add(root.val);
            root = root.right;
        }
    }
    /***
     * 重建二叉树
     *利用前序和中序
     *
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int n = preorder.length;
        for (int i = 0; i < n; i++) {
            indexMap.put(inorder[i], i);
        }
        return myBuildProcess(preorder, inorder, 0, n-1, 0, n-1);
    }

    HashMap<Integer, Integer> indexMap = new HashMap<>();
    private TreeNode myBuildProcess(int[] preorder, int[] inorder, int preorder_left, int preorder_right, int inorder_left, int inorder_right) {
        if (preorder_left > preorder_right){
            return null;
        }
        // 第一个肯定是根节点
        int preorderRoot = preorder_left;
        //中序遍历中的根节点下标
        int root_index = indexMap.get(preorder[preorderRoot]);

        TreeNode root = new TreeNode(preorder[preorderRoot]);
        int size_left = root_index - inorder_left;
        root.left = myBuildProcess(preorder, inorder, preorder_left+1, preorder_left + size_left, inorder_left, inorder_right-1);
        root.right = myBuildProcess(preorder, inorder, preorder_left+size_left+1, preorder_right, root_index +1, inorder_right);
        return root;

    }

    private int[] preorder;
    private HashMap<Integer, Integer> dic = new HashMap<>();
    public TreeNode buildTree1(int[] preorder, int[] inorder) {
        this.preorder = preorder;
        for(int i = 0; i < inorder.length; i++)
            dic.put(inorder[i], i);
        return recur(0, 0, inorder.length - 1);
    }
    private TreeNode recur(int root, int left, int right) {
        if(left > right) return null;                          // 递归终止
        TreeNode node = new TreeNode(preorder[root]);          // 建立根节点
        int i = dic.get(preorder[root]);                       // 划分根节点、左子树、右子树
        node.left = recur(root + 1, left, i - 1);              // 开启左子树递归
        node.right = recur(root + i - left + 1, i + 1, right); // 开启右子树递归
        return node;                                           // 回溯返回根节点
    }

    /***
     * 给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
     *
     * 叶子节点 是指没有子节点的节点。
     * 来源：力扣（LeetCode）
     * 链接：https://leetcode.cn/problems/path-sum-ii
     * 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    List<List<Integer>> ret = new LinkedList<List<Integer>>();
    Deque<Integer> path = new LinkedList<Integer>();

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        pathSumdfs(root, targetSum);
        return ret;
    }
    public void pathSumdfs(TreeNode root, int targetSum) {
        if (root ==null){
            return;
        }
        path.offerLast(root.val);
        targetSum -= root.val;
        if (root.left == null && root.right == null && targetSum ==0){
            ret.add(new LinkedList<Integer>(path));
        }
        pathSumdfs(root.left, targetSum);
        pathSumdfs(root.right, targetSum);
        path.pollLast();
    }

}

    /***
     * 栈实现队列
     *
     * */

class CQueue {
    private LinkedList<Integer> stack1; // 队列入队
    private LinkedList<Integer> stack2; // 队列出队

    public CQueue() {
        stack1 = new LinkedList<>();
        stack2 = new LinkedList<>();
    }

    public void appendTail(int value) {
        stack1.add(value);
    }

    public int deleteHead() {
        if (stack2.isEmpty()){
            if (stack1.isEmpty()){
                return -1;  // 栈都为空时，返回-1
            }
            while (!stack1.isEmpty()) {
                stack2.add(stack1.pop());  // 来回倒腾，使得第一个入栈的在出栈的栈顶
            }
            return stack2.pop();    // 返回队列的第一个元素
        }else return stack2.pop();  // 将整个队列分为两个部分，一部分存着出队元素，一部分存储入队元素
    }
}

class threeSum{
    public static List<List<Integer>> threeSum1(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            List<Integer> threesum = new ArrayList<>();

            if (nums[i] > 0) return res;
            if(i > 0 && nums[i] == nums[i-1]) continue;

            int num = nums[i];
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right){
                if (num + nums[left] + nums[right] == 0 && nums[left] != nums[right]){

                }
            }
        }


        return res;
    }
}









