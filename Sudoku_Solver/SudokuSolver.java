
import java.util.*;

public class SudokuSolver{
	int [][]a;
	public SudokuSolver(int[][] a)
	{
		this.a = a;
	}
	public boolean check(int x,int y,int n)
	{
		boolean f1 = false,f2 = false,f3 = false;
		for(int i=0;i<9;i++)
		{
			if(i!=x&&a[y][i]==n)
			{
				return false;
			}
		}
		f1 = true;
		for(int i=0;i<9;i++)
		{
			if(i!=y&&a[i][x]==n)
			{
				return false;
			}
		}
		f2 = true;
		int s,r;
		s = x/3;
		s = s*3;
		r = y/3;
		r = r*3;
		for(int i=s;i<s+3;i++)
		{
			for(int j=r;j<r+3;j++)
			{
				if(a[j][i]==n&&(i!=x&&j!=y))
				{
					return false;
				}
			}
		}
		f3 = true;
		return f1&&f2&&f3;
	}
	public boolean solve(int x,int y)
	{
		boolean f = false;
		if(a[x][y]==0)
		{
		for(int i=1;i<=9;i++)
		{
			a[x][y]=i;
			if(check(y,x,i))
			{
				a[x][y]=i;
				if(x==8&&y==8)
				{
					return true;
				}
				else if(x==8)
				{
					f = solve(0,y+1);
				}
				else
				{
					f = solve(x+1,y);
				}
			}
			if(f)
			{
				return true;
			}
			else
			{
				a[x][y]=0;
			}
		}
		}
		else
		{
			if(x==8&&y==8)
			{
				return true;
			}
			else if(x==8)
			{
				f = solve(0,y+1);
			}
			else
			{
				f = solve(x+1,y);
			}
			if(f)
			{
				return true;
			}
		}
		return false;
	}
	public int[][] getArray()
	{
		return this.a;
	}
	public static void main(String args[])
	{
		int a[][]=new int[9][9];
		int k;
		Scanner sc = new Scanner(System.in);
		for(int i=0;i<9;i++)
		{
			for(int j=0;j<9;j++)
			{
				k = sc.nextInt();
				a[i][j] = k;
			}
		}
		SudokuSolver dummy = new SudokuSolver(a);
		dummy.solve(0,0);
		a = dummy.getArray();
		System.out.println("Solution is"+'\n');
		for(int i=0;i<9;i++)
		{
			for(int j=0;j<9;j++)
			{
				System.out.print(a[i][j]+" ");
			}
			System.out.println();
		}
	}
}
