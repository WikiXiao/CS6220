package top100words_2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import top100words_2.Pair;

public class Top100Words_2 {

	  public static List<Pair>arr = new ArrayList<Pair>();
	  static Map<String,Integer>mp = new HashMap<String,Integer>();
	  public static class TokenizerMapper
	       extends Mapper<Object, Text, Text, IntWritable>{

	    private final static IntWritable one = new IntWritable(1);
	    private Text word = new Text();

	    public void map(Object key, Text value, Context context
	                    ) throws IOException, InterruptedException {
	      StringTokenizer itr = new StringTokenizer(value.toString());
	      while (itr.hasMoreTokens()) {
	        word.set(itr.nextToken());
	        context.write(word, one);
	      }
	    }
	  }

	  public static class IntSumReducer
	       extends Reducer<Text,IntWritable,Text,IntWritable> {
	    private IntWritable result = new IntWritable();

	    public void reduce(Text key, Iterable<IntWritable> values,
	                       Context context
	                       ) throws IOException, InterruptedException {
	      int sum = 0;
	      for (IntWritable val : values) {
	        sum += val.get();
	      }
	      result.set(sum);
	      Integer tmp = mp.getOrDefault(key.toString(),0);
	      mp.put(key.toString(),tmp+sum);
	      //context.write(key, result);
	    }
	    @Override
	    public void cleanup(Context context) throws IOException, InterruptedException {
	    	for(Map.Entry<String, Integer>entry:mp.entrySet()) {
	    		arr.add(new Pair(entry.getKey(),entry.getValue()));
	    	}
	    	int n = arr.size();
	    	for(int i=0;i<n;++i) {
	    		for(int j=i+1;j<n;++j) {
	    			Pair p1 = arr.get(i);
	    			Pair p2 = arr.get(j);
	    			if(p1.value<p2.value) {
	    				Collections.swap(arr, i, j);
	    			}
	    		}
	    	}
			for(int i=0;i<100;++i) {
				Pair p = arr.get(i);
				context.write(new Text(p.str),new IntWritable(p.value));
			}
	    }
	  }

	  public static void main(String[] args) throws Exception {
	    Configuration conf = new Configuration();
	    Job job = Job.getInstance(conf, "top 100 words");
	    
	    FileInputFormat.addInputPath(job, new Path(args[0]));
	    FileOutputFormat.setOutputPath(job, new Path(args[1]));
	    
	    job.setJarByClass(Top100Words_2.class);
	    job.setMapperClass(TokenizerMapper.class);
	    job.setCombinerClass(IntSumReducer.class);
	    job.setReducerClass(IntSumReducer.class);
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(IntWritable.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(IntWritable.class);
	    System.exit(job.waitForCompletion(true) ? 0 : 1);
	  }
	}
