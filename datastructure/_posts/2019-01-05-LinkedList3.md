## LinkedList 3

### 1. 1 중첩 클래스

Java의 클래스 멤버는 필드, 생성자, 메소드, 인터페이스, 또 다른 클래스 등이다. 다른 클래의 멤버인 클래스를 중첩 클래스(nested class)라고 한다. 클래스 Y가 사용될 것이 어떤 클래스 내부인 경우에 클래스 Y는 사용될 클래스 안에 중첩되어야 한다. 이렇게 해야 Y 클래스가 다른 클래스에서 접근할 수 없는 보안성을 얻게 된다. 

pirvate 중첩 클래의 모든 멤버들은 바깥쪽 클래스의 모든 위치로부터 접근 가능하므로, 단순성을 위해 대개 접근 수정자(private protected, public)를 생략하고 선언한다. 보통의 경우, 중첩 클래스는 그 객체가 바깥쪽 클래스의 비-static 멤버를 접근할 필요가 있지 않는 한, static으로 선언되어야 한다.

```java
public class Main {
    private int m = 22;
    
    public Main() {
        Nested nested = new Nested();
        System.out.println("Outside of Nested; nested.n = " + nested.n);
        nested.f();
    }
    
    private static class Nested {
        private int n = 44;
        
        private void f() {
            System.out.println("Inside of Nested; m = " + m);
        }
    }   
}
```



Node 클래스를 LinkedList 클래스 안에 숨기는 일은, LinkedList 클래스를 캡슐화시켜 이 클래스를 독립적으로 만들고 그 구현 세부 사항을 숨겨주게 된다. 개발자는 그 클래스 외부의 아무런 코드도 수정할 필요 없이 그 구현을 변경 할 수 있다. 

```java
public class LikedList {
    private Node start;
    public void insert(int x) {
        if(start == null || start.data > x) {
            start = new Node(x, start);
            return start;
        }
        Node p = start;
        whlie(p.next != null) {
            if(p.next.data > x) break;
            p = p.next;
        }
        p.next = new Node(x, p.next);
    }
    public void delete(int x) {
        if(start == null || start.data > x)
    		return start;
    	if(start.data == x)
    		return start.next;
    	for(Node p = start; p.next != null; p = p.next) {
            if(p.next.data > x) break;
            if(p.next.data == x) {
                p.next = p.next.next;
                break;
            }
    	}
    }
    private static class Node {
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
}
```



