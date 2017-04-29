/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tools;

import core.game.Observation;
import core.game.StateObservation;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Observable;

import ontology.Types;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

/**
 *
 * @author yuy
 */
public class Recorder {
    public FileWriter filewriter;
    public static Instances s_datasetHeader = datasetHeader();
    
    public Recorder(String filename) throws Exception{
        
        filewriter = new FileWriter(filename+".arff");
        filewriter.write(s_datasetHeader.toString());
        /*
        // ARFF File header
        filewriter.write("@RELATION AliensData\n");
        // Each row denotes the feature attribute
        // In this demo, the features have four dimensions.
        filewriter.write("@ATTRIBUTE gameScore  NUMERIC\n");
        filewriter.write("@ATTRIBUTE avatarSpeed  NUMERIC\n");
        filewriter.write("@ATTRIBUTE avatarHealthPoints NUMERIC\n");
        filewriter.write("@ATTRIBUTE avatarType NUMERIC\n");
        // objects
        for(int y=0; y<14; y++)
            for(int x=0; x<32; x++)
                filewriter.write("@ATTRIBUTE object_at_position_x=" + x + "_y=" + y + " NUMERIC\n");
        // The last row of the ARFF header stands for the classes
        filewriter.write("@ATTRIBUTE Class {0,1,2}\n");
        // The data will recorded in the following.
        filewriter.write("@Data\n");*/
        
    }


    /**
     * itype对应关系:
     * 0:墙壁,1:玩家,2:障碍,5:炸弹，6:敌人(NPC)
     * Immovable:0,2;  Movable:5;  NPC:6
     */
    public static double[] featureExtract(StateObservation obs){
        double[] feature = new double[14*3+2+1+1];  // 14*3 + 2(NPC) + 1(hasBomb) + 1(class)
        Vector2d avatarPos=obs.getAvatarPosition();
        int avatarPos_X = (int)(avatarPos.x/25);
        int left_NPC_number=0;
        int right_NPC_number=0;
        boolean hasBomb=false;
        
        // 42 locations
        int[][] map = new int[3][14];
        // Extract features
        LinkedList<Observation> allobj = new LinkedList<>();
        if( obs.getImmovablePositions()!=null ) {
            for (ArrayList<Observation> l : obs.getImmovablePositions()) allobj.addAll(l);
        }
        if( obs.getMovablePositions()!=null ) {
            for (ArrayList<Observation> l : obs.getMovablePositions()) allobj.addAll(l);
        }
        if( obs.getNPCPositions()!=null ) {
            for (ArrayList<Observation> l : obs.getNPCPositions()) {
                for(Observation o:l){
                    int NPC_X=(int)(o.position.x/25);
                    if(NPC_X<avatarPos_X)
                        left_NPC_number++;
                    else if(NPC_X>avatarPos_X)
                        right_NPC_number++;
                }
                allobj.addAll(l);
            }
        }

        for(Observation o : allobj){
            Vector2d p = o.position;
            int x = (int)(p.x/25);
            int y= (int)(p.y/25);
            if(Math.abs(x-avatarPos_X)<=1)
                map[x-avatarPos_X+1][y] = o.itype;
            if(o.itype==5 && Math.abs(p.x-avatarPos.x)<25)//被炸弹炸到
                hasBomb=true;
        }
        for(int y=0; y<14; y++)
            for(int x=0; x<3; x++)
                feature[y*3+x] = map[x][y];

        //left and right NPC number
        feature[42]=left_NPC_number;
        feature[43]=right_NPC_number;
        feature[44] = hasBomb ? 10.0 : 0.0;
        
        return feature;
    }

    /*
    public static double[] featureExtract(StateObservation obs){
        double[] feature = new double[453];  // 448 + 4 + 1(class)
        //Vector2d avatarPos=obs.getAvatarPosition();
        //boolean hasBomb=false;

        // 448 locations
        int[][] map = new int[32][14];
        // Extract features
        LinkedList<Observation> allobj = new LinkedList<>();
        if( obs.getImmovablePositions()!=null ) {
            //System.out.println("immovable");
            for (ArrayList<Observation> l : obs.getImmovablePositions()) allobj.addAll(l);
        }
        if( obs.getMovablePositions()!=null ) {
            //System.out.println("movable");
            for (ArrayList<Observation> l : obs.getMovablePositions()) allobj.addAll(l);
        }
        if( obs.getNPCPositions()!=null ) {
            //System.out.println("NPC\n");
            for (ArrayList<Observation> l : obs.getNPCPositions()) allobj.addAll(l);
        }

        for(Observation o : allobj){
            Vector2d p = o.position;
            int x = (int)(p.x/25);
            int y= (int)(p.y/25);
            map[x][y] = o.itype;
            //if(o.itype==5 && Math.abs(p.x-avatarPos.x)<25)//被炸弹炸到
                //hasBomb=true;
        }
        for(int y=0; y<14; y++)
            for(int x=0; x<32; x++)
                feature[y*32+x] = map[x][y];

        // 4 states
        feature[448] = obs.getGameTick();
        feature[449] = obs.getAvatarSpeed();
        feature[450] = obs.getAvatarHealthPoints();
        feature[451] = obs.getAvatarType();

        //feature[448] = hasBomb ? 10.0 : 0.0;

        //System.out.println(obs.getAvatarType());
        //System.out.println(obs.getAvatarPosition());

        return feature;
    }
    */


    public static Instances datasetHeader(){
        FastVector attInfo = new FastVector();
        // 448 locations
        for(int y=0; y<14; y++){
            for(int x=-1; x<=1; x++){
                Attribute att = new Attribute("object_at_position_x=" + "avatarX"+ x + "_y=" + y);
                attInfo.addElement(att);
            }
        }
        Attribute att;
        //left and right NPC number
        att = new Attribute("left_NPC_number");attInfo.addElement(att);
        att = new Attribute("right_NPC_number");attInfo.addElement(att);
        att = new Attribute("hasBomb" ); attInfo.addElement(att);
        //class
        FastVector classes = new FastVector();
        classes.addElement("0");
        classes.addElement("1");
        classes.addElement("2");
        classes.addElement("3");
        att = new Attribute("class", classes);
        attInfo.addElement(att);

        Instances instances = new Instances("AliensData", attInfo, 0);
        instances.setClassIndex( instances.numAttributes() - 1);

        return instances;
    }

    /*
    public static Instances datasetHeader(){
        FastVector attInfo = new FastVector();
        // 448 locations
        for(int y=0; y<14; y++){
            for(int x=0; x<32; x++){
                Attribute att = new Attribute("object_at_position_x=" + x + "_y=" + y);
                attInfo.addElement(att);
            }
        }
        Attribute att;
        att = new Attribute("GameTick" ); attInfo.addElement(att);
        att = new Attribute("AvatarSpeed" ); attInfo.addElement(att);
        att = new Attribute("AvatarHealthPoints" ); attInfo.addElement(att);
        att = new Attribute("AvatarType" ); attInfo.addElement(att);
        //att = new Attribute("hasBomb" ); attInfo.addElement(att);
        //class
        FastVector classes = new FastVector();
        classes.addElement("0");
        classes.addElement("1");
        classes.addElement("2");
        classes.addElement("3");
        att = new Attribute("class", classes);        
        attInfo.addElement(att);
        
        Instances instances = new Instances("AliensData", attInfo, 0);
        instances.setClassIndex( instances.numAttributes() - 1);
        
        return instances;
    }
    */


    // Record each move as the ARFF instance
    public void invoke(StateObservation obs, Types.ACTIONS action) {
        double[]  feature = featureExtract(obs);

        try{  
            for(int i=0; i<feature.length-1; i++)
                filewriter.write(feature[i] + ",");

            // Recorde the move type as ARFF classes
            int action_num = 0;
            if( Types.ACTIONS.ACTION_NIL == action) action_num = 0;
            if( Types.ACTIONS.ACTION_USE == action) action_num = 1;
            if( Types.ACTIONS.ACTION_LEFT == action) action_num = 2;
            if( Types.ACTIONS.ACTION_RIGHT == action) action_num = 3;
            filewriter.write(action_num + "\n");
            filewriter.flush();
        }catch(Exception exc){
            exc.printStackTrace();
        }
    }
    
    public void close(){
        try{
            filewriter.close();
        }catch(Exception exc){
            exc.printStackTrace();
        }
    }
}
