## Linked List 2

### 1. 1 연결 리스트에 대한 원소 삽입

파라미터 2개의 짜리 생성자를 통하여 노드 생성과 삽입을 한꺼번에 수행할 수 있도록 해준다. 삽입은 새로운 노드 앞에 놓일 리스트 노드 p를 발견하고 새로운 노드를 생성해 부착하는 두 단계로 진행된다. 

```java
class Node {
    int data;
    Node next;
    
    Node(int data) {
        this.data = data;
    }
    
    Node(int data, Node next) {
        this.data = data;
        this.next = next;
    }
}
```

### 공백이 없는 정렬된 노드에 대한 삽입

```java
void insert(Node start, int x) {
    // PRECONDITIONS: the list is in ascending order, and x > start.data;
    // POSTCONDITIONS: the list is in ascending order, and it contains x;
    Node p = start;
    while(p.next != null) { // x보다 큰 데이터에서 빠져나와 p를 가르키게 한다. 
        if(p.next.data > x) break;
        p = p.next; 
    }
    p.next = new Node(x, p.next); // p의 다음을 x노드를 가르키게 만들고 다음 노드를 가르키게 만든다. 
}
```

### 첫번째 노드에 삽입

```java
Node insert(Node start, int x) {
    // precondition: the list is in ascending order;
    // postcondition: the list is in ascending order, and it contains x;
    // 첫번쨰 노드에 삽입하는 부분.
    if(start == null || start.data > x) {
        start = new Node(x.start);
        return start;
    }
    
    Node p = start;
    while (p.next != null) {
		if(p.next.data > x) break;
		p = p.next;
	}
	p.next = new Node(x, p.next);
	return start;
}
```

### 1. 2 연결 리스트에서의 삭제

insert메소드와 마찬가지로 delete() 메소드도 원소를 발견하고, 삭제하는 두 단계로 진행된다. 

```java
Node delete(Node start, int x) {
    // precondition: the list is in ascending order;
    // postcondition: the list is in ascending order, and if it did
    // contains x, then the first occurrence of x has been deleted;
    if(start == null || start.data > x) // x is not in the list
    	return start;
    if(start.data == x) // x is the first element in the list
    	return start.next;
    for(Node p = start; p.next != null; p = p.next) {
        if(p.next.data > x) break; // x is not in the list
        if(p.next.data == x) { // x is in the p.next node
            p.next = p.next.next; // delete it
            break;
        }
    }
    return start;
}
```