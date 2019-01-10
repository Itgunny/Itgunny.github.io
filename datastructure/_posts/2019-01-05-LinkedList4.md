## Linked List 4

### BigInteger로 보는 Linked List

자바에서는 19 자리 이상의 정수가 필요할 경우, 임의의 길이의 정수를 허용하는 java.math.BigInteger 클래스를 사용하면 된다. 아래는  LinkedList를 통하여 어떻게 표현되는 지 보여주는 클래스이다. 

BigInt x = new BigInt(13579);

위 함수를 예를 들면 처음 n /= 10을 만나 9에 해당되는 노드가 생성된다. 노드를 넣어 주고 1357이 된다. 135 -> Node(7); / 13 -> Node(5); / 1 -> Node(3) / Node(1) 이 되므로 모든 노드가 완성된다. 

덧셈은 다음과 같은 순서로 이루어 진다.

1. 주어진 두 리스트의 다음 자리들을 n의 10의 자리에 더한 값을 n에 대체.
2. n의 1의 자리를 포함하는 새 Node를 z에 첨가. 
3. 모든 3개 포인터 p, q, r을 전진.

```java
public class BigInt {
    private Node start;
    
    public BigInt(int n) {
		if( n < 0 ) throw new IllegalArgumentException(n + "<0");
		start = new Node(n % 10);
		Node p = start;
		n /= 10;
		while (n > 0) {
			p = p.next = new Node(n % 10);
			n /= 10;
		}
	}
    
    public BigInt(String s) {
		if(s.length() == 0)
			throw new IllegalArgumentException("empty string");
		start = new Node(digit(s, s.length() - 1));
		Node p = start;
		for(int i = s.length() - 2; i >= 0; i--)
			p = p.next = new Node(digit(s, i))
	}
	
	private int digit(String s, int i) {
		String ss = s.substring(i, i+1);
		return Integer.parseInt(ss);
	}
    
    public BigInt plus(BigInt y) {
		Node p = start, q = y.start;
        int n = p.digit + q.digit;
        BigInt z = new Big(n % 10);
        Node r = z.start;
        p = p.next;
        q = q.next;
        
        while(p != null & q != null) {
            n = n/10 + p.digit + q.digit;
            r.next = new Node(n % 10);
            p = p.next;
            q = q.next;
            r = r.next;
        }
        
        while(p != null) {
            n = n/10 + p.digit;
            r.next = new Node(n % 10);
            p = p.next;
            r = r.next;
        }
        
        while(q != null) {
            n = n/10 + q.digit;
            r.next = new Node(n % 10);
            q = q.next;
            r = r.next;
        }
        
        if(n > 9) r.next = new Node(n / 10);
        return z;
    }
    
    public String toString() {
        StringBuffer buf = new StringBuffer(Integer.toString(start.digit));
        Node p = start.next;
        while(p != null) {
            buf.insert(0, Integer.toString(p.digit)) {
                p = p.next;
            }
        }
        return buf.toString();
    }
    
    private static class Node {
        int digit;
        Node next;
        Node(int digit) { this.digit = digit;}
    }
}
```

