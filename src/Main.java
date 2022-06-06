import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class Main {

    public static void main(String[] args) {

        ArrayList<data> dataList = readDataFile("app1.data");
        ArrayList<data> testList = readDataFile("app1.test");

        dataList = fillUnknownValues(dataList);
        testList = fillUnknownValues(testList);

        dataList = normalizeDataList(dataList);
        testList = normalizeDataList(testList);

        trainAndClassify(dataList,testList,13);
        crossValidation(dataList,testList);
    }

    private static int crossValidation(ArrayList<data> dataList, ArrayList<data> testList){
        System.out.println("Start Crossvalidation");
        ArrayList<data> combinedData = dataList;
        combinedData.addAll(testList);

        int ymin = 1;
        int ymax = combinedData.size()-1;
        double[] min_error = {ymin,combinedData.size()};//First Value: k  Second Value: Errorvalue
        for (int k = ymin; k <= ymax; k++) {
            System.out.print("\nCrossvalidation: Checking k="+k);
            double err_count = 0;
            for (data dataset:combinedData) {
                ArrayList<data> trainData = (ArrayList<data>) combinedData.clone();
                trainData.remove(dataset);

                if(!classify(dataset,trainData,k))
                    err_count++;
            }
            double avg_error = err_count/(combinedData.size()-1);
            if(avg_error < min_error[1]){
                min_error[0] = k;
                min_error[1] = avg_error;
            }
            System.out.print(" Average Error:"+avg_error);
        }
        System.out.println("\nLowest error Value with K="+min_error[0]+" with Avg-Error="+min_error[1]);
        return (int) min_error[0];
    }

    private static int[] trainAndClassify(ArrayList<data> trainData, ArrayList<data> testData, int k){
        System.out.println("Start Classifing with k="+k);
        int correctcount = 0;
        int falsecount = 0;
        for (data dataset:testData) {
            if(classify(dataset, trainData, k)) correctcount++;
            else falsecount++;
        }
        double error_rate = ((double)falsecount/((double)correctcount+(double)falsecount));
        System.out.println("\nResult:\Correctly classified:"+correctcount+" | Incorrectly classified:"+falsecount+" | Error Rate:"+ error_rate);
        return new int[]{correctcount, falsecount};
    }

    private static ArrayList<data> fillUnknownValues(ArrayList<data> dataList){
        for (data dataset:dataList) {
            int index = 0;
            for (double value: dataset.getValues()) {
                if ((int) value == -1) {
                    dataset.setValue(getMostCommonAttrValue(dataList,index,dataset.getKlasse()),index);
                }
                index++;
            }
        }
        return dataList;
    }

    private static ArrayList<data> readDataFile(String fileName){
        ArrayList<data> dataList = new ArrayList<>();
        try(BufferedReader in = new BufferedReader(new FileReader(fileName))) {
            String str;
            while ((str = in.readLine()) != null) {
                String[] tokens = str.split(",");
                double[] dataBufferArray = new double[tokens.length];
                int i=0;
                for (String token:tokens) {
                    try {
                        dataBufferArray[i] = Double.parseDouble(token);
                    }
                    catch (NumberFormatException e)
                    {
                        dataBufferArray[i] = -1;
                    }
                    i++;
                }
                dataList.add(new data(dataBufferArray));
            }
        }
        catch (IOException e) {
            System.out.println("File Read Error");
        }
        return dataList;
    }

    //Returns the Most Common Value an Attribute has in the given Dataset
    private static double getMostCommonAttrValue(ArrayList<data> dataList, int columnindex, int klasse){
        ArrayList<Double> tempList = new ArrayList<>();

        for (data data:dataList) {
            if(data.getKlasse() == klasse)
                tempList.add(data.getValue(columnindex));
        }
        double[] max_occ = {0,0};//First Element: Value  Second Element: occurence
        for (double value:tempList) {
            if((int) value != -1) {
                int occ = Collections.frequency(tempList, value);
                if (occ > max_occ[1]) {
                    max_occ[0] = value;
                    max_occ[1] = occ;
                }
            }
        }
        return max_occ[0];
    }

    private static ArrayList<data> normalizeDataList(ArrayList<data> dataList)
    {
        int size = dataList.get(0).getValuesSize();
        double[] max = new double[size];
        double[] min = new double[size];
        for (int i = 0; i < size; i++) {
            double[] maxmin = getMaxMinAttrValue(dataList,i);
            max[i] = maxmin[0];
            min[i] = maxmin[1];
        }
        for (data dataset:dataList) {
            int index = 0;
            for (double value:dataset.getValues()) {
                double normalized = (value - min[index])/(max[index]- min[index]);
                dataset.setValue(normalized,index);
                index++;
            }
        }
        return dataList;
    }

    private static double[] getMaxMinAttrValue(ArrayList<data> dataList, int columnindex){
        ArrayList<Double> tempList = new ArrayList<>();

        for (data data:dataList) {
            tempList.add(data.getValue(columnindex));
        }
        double[] maxmin_value = {tempList.get(0),tempList.get(0)};//First Element: Max  Second Element: Min

        for (double value:tempList) {
            if((int) value > maxmin_value[0]) {
                maxmin_value[0] = value;
            }
            if((int) value < maxmin_value[1])
            {
                maxmin_value[1] = value;
            }
        }
        return maxmin_value;
    }

    private static boolean classify(data testdataset, ArrayList<data> dataList, int k){
        double[] to_classify = testdataset.getValues();
        dataList.get(0).setTo_classify(to_classify);
        dataList.forEach(data -> data.calcDistance());
        dataList.sort(data::compareTo);
        int result = kNearestNeigbour(dataList, k);

        //System.out.print("\nZu klassifizieren:");
        //testdataset.printData();
        if(result == testdataset.getKlasse()) {
            //System.out.println("\nKorrekt klassifiziert als " + result);
            return true;
        }
        else {
            //System.out.println("\nFalsch klassifiziert als " + result);
            return false;
        }
    }

    private static int kNearestNeigbour(ArrayList<data> list, int k){
        if(k > list.size())
            throw new Error("Error: k to big");
        int max_class = 0;
        for(int i = 0;i<k;i++){
            int klasse = list.get(i).getKlasse();
            if(klasse > max_class)
                max_class = klasse;
        }
        ArrayList<Integer> counter = new ArrayList<>(Arrays.asList(new Integer[max_class+1]));
        Collections.fill(counter,0);
        for(int i = 0;i<k;i++){
            int klasse = list.get(i).getKlasse();
            counter.set(klasse,counter.get(klasse)+1);
        }
        int max = Collections.max(counter);
        return counter.indexOf(max);
    }

    private static void printList(ArrayList<data> list){
        for (data i:list) {
            i.printData();
        }
    }


}

class data implements Comparable<data>{
    private double[] values;
    private double distance;
    private static double[] to_classify;

    public data(double[] values ) {
        this.values = values;
    }

    public double[] getValues(){
        return this.values;
    }

    public double getValue(int index)
    {
        return this.values[index];
    }

    public int getValuesSize(){
        return this.values.length;
    }

    public void setValue(double value , int index)
    {
        this.values[index] = value;
    }

    public int getKlasse() {
        return (int) values[values.length-1];
    }

    public static void setTo_classify(double[] to_classify) {
        data.to_classify = to_classify;
    }

    @Override
    public int compareTo(data data) {
        double dist = data.calcDistance();
        if(this.distance == dist)
            return 0;
        else if(this.distance > dist)
            return 1;
        else
            return -1;
    }

    public void printData(){
        System.out.printf("\n|Values: ");
        for (double i:this.values) {
            System.out.printf("%1.2f  ",i);
        }
        System.out.printf(" | Class: %d | Distance: %1.2f |",this.getKlasse(),this.distance);
    }


    public double calcDistance(){
        double dist = 0;
        if(this.values.length != to_classify.length)
            throw new Error("Error: Variable Length doesn't match");
        for(int i = 0; i<this.values.length-1;i++)
            dist = dist + Math.abs(this.values[i] - to_classify[i]);

        this.distance = dist;
        return dist;
    }


}
