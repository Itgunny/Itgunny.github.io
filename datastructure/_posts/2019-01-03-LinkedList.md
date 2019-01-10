## Linked List

### 1. 1 순서가 있는 배열의 유지

이진 탐색은 배열은 정렬된 순서로 유지해야 한다. 정렬된 배열에 새 원소를 삽입하는 것은 어렵다. 배열의 원소를 넣기 위해선 그보다 큰 원소를 모두 이동시켜야 된다는 단점이 있다. 

아래와 같이 배열은 insert 하면서 배열을 이동 시켜줘야 한다는 단점이 있다. 

```java
int[] insert(int[] a, int n, int x) {
    // preconditions: [0] <= ... <= a[n-1], and n < a.length;
    // postconditions: a[0] <= ... <= a[n], and x is among them;
    int i = 0; // find the first index i for which a[i] > x:
    while(i < n && a[i] <= x) {
        ++i;
    }
    // shift {a[i], ..., a[n-1]} into {a[i+1], ..., a[n]}:
    System.arraycopy(a, i, a, i+1, n-i);
    // insert x into a[i]:
    a[i] = x;
}
```

### 1. 2 간접 참조

데이터 이동 문제에 대한 해결방법은 원소의 실제 위치를 추적하기 위하여 인덱스 배열을 사용하는 것이다. 이 해결법은 추가의 공간을 필요로 하고 코드를 복잡하게 만들지만 원소를 이동 시킬 필요성을 제거한다. 

 ```java
void insert(int[] a, int[] k, int x, int free) {
    int i = 0;
    while(k[i] != 0 && a[k[i]] < x)
    	i = k[i];
    a[free] = x; // x를 다음의 자유 위치에 삽입
    k[free] = k[i]; // 그 위치의 다음 인덱스를 k[]에 저장
    k[i] = free++; // x의 인덱스를 k[]에 저장 -> 자유 위치를 다음의 비사용 위치로 전진
}
 ```

### 1. 3 연결 노드

배열을 따로 인덱스로 만들 필요 없이 Node 클래스는 자기 참조(self-referential) 클래스이다. next 필드가 타입 Node로 선언되어 있다. 각 Node 객체는 Node 객체에 대한 참조를 갖는 필드를 포함하고 있다. 참조는 다른 객체의 메모리 주소값을 참조하여 다음 원소를 가르킨다. C에서는 pointer라는 개념이 있다. Node의 next값을 할당하지 않으면 null값으로 표기 된다. 마지막 참조값은 next를 기준으로 null 을 이용하여 판단할 수 있다. 

```java
class Node {
    private int data;
    private Node next;
    
    public Node(int data) {
        this.data = data;
    }
    
    public static void Main(String args[]) {
        Node start = new Node(22);
        start.next = new Node(33);
        start.next.next = new Node(44);
        start.next.next.next = new Node(55);
        start.next.next.next.next = new Node(66);
        
        Node p = start;
        p.next = new Node(33); // p의 next에 Node 할당
        p = p.next; // p를 p.next로 이동
        p.next = new Node(44);
        p = p.next;
        p.next = new Node(55);
        p = p.next;
        p.next = new Node(66);
        
        for(Node p = start; p!= null; p = p.next) {
            System.out.println(p.data);
        }
	}
}

```

