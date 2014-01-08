package com.example.segmentation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import android.os.Bundle;
import android.os.SystemClock;
import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.WindowManager;
import android.widget.ImageView;

public class OCRCamera extends Activity implements CvCameraViewListener {

	String TAG = "OCRCamera Module";
	private ImageView mimageView;
	private CameraBridgeViewBase mOpenCvCameraView;
	private Bitmap mBitmapPreview;
	private int indice;
	private int topindice;
	private long mActTime, mElaTime;
	private boolean mFirstTime;
	private org.opencv.core.Mat mauxiliar;
	private List<MatOfPoint> mcontours;
	private MatOfPoint2f mMOP2f1;
	private Mat mhierarchy;
	private MatOfPoint mcontour;
	private Scalar mblanco;
	private Scalar mnegro;
	private org.opencv.core.Mat mderivate,mhorizontal, mvertical;
	private Moments mMom;
	private ArrayList<Point> mLppp;

	@Override
	public void onCreate(Bundle savedInstanceState) {
	    Log.i(TAG, "called onCreate");
	    super.onCreate(savedInstanceState);
	    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
	    setContentView(R.layout.activity_ocrcamera);
	    
	    
	    //Para medir el tiempo 
	    mActTime = SystemClock.elapsedRealtime();
	    
	    
	    mFirstTime = true;
	    topindice = 11;
	    indice = 10; //Indice de la imagen
	    
	    
	    
	    
	    //Iniciar Interfaz
	    mimageView = (ImageView) findViewById(R.id.imageView);
	    
	    mimageView.setOnTouchListener(new OnTouchListener() {
			public boolean onTouch(View v, MotionEvent event) {
				indice++;
				if(indice == topindice) indice = 0;
				runOnUiThread(new Runnable() {

				    public void run() {
				    	mimageView.setImageBitmap(mBitmapPreview);
				    	mimageView.setBackgroundColor(Color.BLACK);
				    }
				});
				return false;
			}
		});
	    
	    mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.HelloOpenCvView);
	    mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
	    mOpenCvCameraView.setCvCameraViewListener(this);
	    //Fin inicio de interfaz
	    
	    
	    
	}
	
	@Override
	public void onPause()
	{
	    super.onPause();
	    if (mOpenCvCameraView != null)
	        mOpenCvCameraView.disableView();
	}
	
	public void onDestroy() {
	    super.onDestroy();
	    if (mOpenCvCameraView != null)
	        mOpenCvCameraView.disableView();
	}
	
	public void onCameraViewStarted(int width, int height) {
	}
	
	public void onCameraViewStopped() {
	}
	
	@Override
	public Mat onCameraFrame(Mat inputFrame) {
		
		//Iniciar bitmaps
		if(mFirstTime){
			//Iniciar variables camara
		    mauxiliar = new Mat();
			mcontours = new ArrayList<MatOfPoint>();
			mMOP2f1 = new MatOfPoint2f();
			mhierarchy = new Mat();
			mblanco = new Scalar(255,255,255);
			mnegro = new Scalar(0,0,0);
			mBitmapPreview = Bitmap.createBitmap(inputFrame.cols(),inputFrame.rows(),Bitmap.Config.ARGB_8888);
			mderivate= new Mat(new Size(inputFrame.cols(),inputFrame.rows()),inputFrame.type());
			mFirstTime = false;
			mLppp  = new ArrayList<Point>();
		}
		
		
		
		//Para medir el tiempo 
	    mElaTime = SystemClock.elapsedRealtime();
	    
	    if(mElaTime >= mActTime + 3000)
	    {
	    	//Actulizo el tiempo
	    	mActTime = mElaTime;
			
			
			mauxiliar = inputFrame.clone();	
			
			
			
			
			
			Imgproc.cvtColor(inputFrame, mauxiliar, Imgproc.COLOR_BGR2GRAY);
			//Debug
			if(indice == 0)
				Utils.matToBitmap(mauxiliar, mBitmapPreview); 
			Imgproc.GaussianBlur(mauxiliar,mauxiliar, new Size(7,7),0);
			//Debug
			if(indice == 1)
				Utils.matToBitmap(mauxiliar, mBitmapPreview); 
			Imgproc.adaptiveThreshold(mauxiliar, mauxiliar, 255, 
					Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);
			//Debug
			if(indice == 2)
				Utils.matToBitmap(mauxiliar, mBitmapPreview); 
			
			
			
			
			mvertical = new Mat(new Size(mauxiliar.cols(),mauxiliar.rows()),mauxiliar.type(),mnegro);
			mhorizontal = new Mat(new Size(mauxiliar.cols(),mauxiliar.rows()),mauxiliar.type(),mnegro);
			
			
			
			//Core.divide(auxiliar,auxiliar2,auxiliar);
			//Core.normalize(auxiliar, auxiliar, 0, 255, Core.NORM_MINMAX);
			
			
			//Vertical lines
			Imgproc.Sobel(mauxiliar, mderivate, CvType.CV_8U, 1, 0, 3, 1, 0);
			//Debug
			if(indice == 3)
				Utils.matToBitmap(mderivate, mBitmapPreview); 
			
			Imgproc.morphologyEx(mderivate,mderivate,Imgproc.MORPH_DILATE,
							Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(3,10)));
			
			//Debug
			if(indice == 4)
				Utils.matToBitmap(mderivate, mBitmapPreview); 
			
			Imgproc.findContours(mderivate, mcontours, mhierarchy, Imgproc.RETR_EXTERNAL, 
					Imgproc.CHAIN_APPROX_SIMPLE);
			
			for(int i=0;i<mcontours.size();++i)
			{
				mcontour = mcontours.get(i);
				
				Rect boundRect = Imgproc.boundingRect(mcontour);
				if(boundRect.height > 300)
					Imgproc.drawContours(mvertical, mcontours, i, mblanco, 4);
				//else
				//	Imgproc.drawContours(auxiliar, contours, i, negro, 4);
				/*contourarea = Imgproc.contourArea(contour);
				contours.get(i).convertTo(mMOP2f1, CvType.CV_32FC2);
				peri = Imgproc.arcLength(mMOP2f1, true);
				Imgproc.approxPolyDP(mMOP2f1,mMOP2f1,0.02*peri,true);
				if(contourarea > max_area &&  mMOP2f1.cols() == 4) 
				{
					max_area = contourarea;
					biggest = i;
				}*/
			}
			
			//Debug
			if(indice == 5)
				Utils.matToBitmap(mvertical, mBitmapPreview); 
			
			mcontours.clear();
			//End Vertical lines
			
			
			//Horizontal lines
			Imgproc.Sobel(mauxiliar, mderivate, CvType.CV_8U, 0, 1, 3, 1, 0);
			//Debug
			if(indice == 6)
				Utils.matToBitmap(mderivate, mBitmapPreview); 
			
			Imgproc.morphologyEx(mderivate,mderivate,Imgproc.MORPH_DILATE,
							Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(10,3)));
			//Debug
			if(indice == 7)
				Utils.matToBitmap(mderivate, mBitmapPreview); 
			
			Imgproc.findContours(mderivate, mcontours, mhierarchy, Imgproc.RETR_EXTERNAL, 
					Imgproc.CHAIN_APPROX_SIMPLE);
			
			for(int i=0;i<mcontours.size();++i)
			{
				mcontour = mcontours.get(i);
				
				Rect boundRect = Imgproc.boundingRect(mcontour);
				if(boundRect.width > 300)
					Imgproc.drawContours(mhorizontal, mcontours, i, mblanco, 4);
			}
			
			//Debug
			if(indice == 8)
				Utils.matToBitmap(mhorizontal, mBitmapPreview);
			
			mcontours.clear();
			//End Horizontal lines
			
			//Los puntos de union de las lineas verticales y horizontales son los puntos de union
			Core.bitwise_and(mvertical, mhorizontal, mauxiliar);
			//Debug
			if(indice == 9)
				Utils.matToBitmap(mauxiliar, mBitmapPreview);
			
			Imgproc.morphologyEx(mauxiliar,mauxiliar,Imgproc.MORPH_DILATE,
					Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(10,10)));
			//Debug
			if(indice == 10)
				Utils.matToBitmap(mauxiliar, mBitmapPreview);
			
			Imgproc.findContours(mauxiliar, mcontours, mhierarchy, Imgproc.RETR_LIST, 
					Imgproc.CHAIN_APPROX_SIMPLE);
			
			mLppp.clear();
			
			for(int i=0;i<mcontours.size();++i)
			{
				mcontour = mcontours.get(i);
				mMom = Imgproc.moments(mcontour);
				Point ppp = new Point(mMom.get_m10()/mMom.get_m00(),mMom.get_m01()/mMom.get_m00());
				mLppp.add(ppp);
				Core.circle(inputFrame, ppp, 4, new Scalar(0,255,0));
			}
			
			
			Arrays.sort(mLppp,new Comparator<Point>() {
				@Override
				public int compare(Point lhs, Point rhs) {
					// TODO Auto-generated method stub
					return 0;
				}
			});
			
			mcontours.clear();
			/*auxiliar.convertTo(auxiliar, CvType.CV_8UC4);
			Scalar s = new Scalar(255,0,0);
			if(biggest != -1)
				Imgproc.drawContours(auxiliar, contours, biggest, s, 4);*/
            
            
					
			//Imgproc.Sobel(auxiliar, auxiliar, CvType.CV_8U, 1, 0,3,1,0);
			//Imgproc.threshold(auxiliar, auxiliar, 0, 255, Imgproc.THRESH_OTSU|Imgproc.THRESH_BINARY_INV);
					
			//Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(64, 35));
			//Imgproc.morphologyEx(auxiliar, auxiliar, Imgproc.MORPH_CLOSE, element);
			
			
			//Imgproc.medianBlur(auxiliar,auxiliar, 5);
			//Utils.matToBitmap(auxiliar, mBitmapPreview[1]); //Imagen blurred
			//Imgproc.Laplacian(auxiliar, auxiliar, CvType.CV_8U);
			//Utils.matToBitmap(auxiliar, mBitmapPreview[2]); //Imagen Laplacian
			//Mat otra = new Mat();
			//Imgproc.threshold(auxiliar, otra, -1, 255, 
			//	    Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);
			//Utils.matToBitmap(otra, mBitmapPreview[3]); //Imagen Threshold
			
			
			
			//Imagen final
			
			
			//Actualizar los Bitmaps
			runOnUiThread(new Runnable() {
			    public void run() {
			    	mimageView.setImageBitmap(mBitmapPreview);
			    	mimageView.setBackgroundColor(Color.BLACK);
			    }
			});
	    }

	    
	    for(int i=0;i<mLppp.size();++i)
			Core.circle(inputFrame, mLppp.get(i), 4, new Scalar(0,255,0));
	    
	    return inputFrame;
	}
	
	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
		return inputFrame.rgba();
	}
	 
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_6, this, mLoaderCallback);
    }

    
    public int compare(){
		return 0;
    	
    }
}
