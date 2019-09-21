package top100words_1;

import java.util.Comparator;

@SuppressWarnings("rawtypes")
public class MyComparator implements Comparator {
	@Override
	public int compare(Object o1, Object o2) {
		// TODO Auto-generated method stub
		Pair p1 = (Pair)o1;
		Pair p2 = (Pair)o2;
		if(p1.value<p2.value) {
			return 1;
		}
		else if(p1.value==p2.value) {
			return 0;
		}
		else return -1;
	}
}
