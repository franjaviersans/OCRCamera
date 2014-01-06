package com.example.segmentation;

import java.util.ArrayList;
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
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

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
	private Bitmap mBitmapPreview[];
	private int indice;
	private int topindice;
	private long mActTime, mElaTime;
	private boolean mFirstTime;

	@Override
	public void onCreate(Bundle savedInstanceState) {
	    Log.i(TAG, "called onCreate");
	    super.onCreate(savedInstanceState);
	    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
	    setContentView(R.layout.activity_ocrcamera);
	    
	    
	    //Para medir el tiempo 
	    mActTime = SystemClock.elapsedRealtime();
	    
	    
	    mFirstTime = true;
	    topindice = 1;
	    indice = 0; //Indice de la imagen
	    mBitmapPreview = new Bitmap[topindice]; //Varias Imagenes de los diferentes pasos
	    
	    
	    
	    
	    //Iniciar Interfaz
	    mimageView = (ImageView) findViewById(R.id.imageView);
	    
	    mimageView.setOnTouchListener(new OnTouchListener() {
			public boolean onTouch(View v, MotionEvent event) {
				indice++;
				if(indice == topindice) indice = 0;
				runOnUiThread(new Runnable() {

				    public void run() {
				    	mimageView.setImageBitmap(mBitmapPreview[indice]);
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
		
		
		
		
		//Para medir el tiempo 
	    mElaTime = SystemClock.elapsedRealtime();
	    
	   // if(mElaTime >= mActTime + 3000)
	    //{
	    	//Actulizo el tiempo
	    	mActTime = mElaTime;
	    	
	    	
			Mat auxiliar = new Mat(), original = inputFrame;
			inputFrame.convertTo(auxiliar, CvType.CV_8U);
			
			//Iniciar todos los bitmaps
			if(mFirstTime)
				for(int i=0; i<topindice; ++i)
					mBitmapPreview[i] = Bitmap.createBitmap(auxiliar.cols(),auxiliar.rows(),Bitmap.Config.ARGB_8888);
			mFirstTime = false;
			
			
			
			Imgproc.cvtColor(inputFrame, auxiliar, Imgproc.COLOR_BGR2GRAY);
			Imgproc.GaussianBlur(auxiliar,auxiliar, new Size(7,7),0);
			Imgproc.adaptiveThreshold(auxiliar, auxiliar, 255, 
					Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);
			
			
			//Imgproc.Sobel(auxiliar, auxiliar, CvType.CV_8U, 1, 0, 3, 1, 0);
			Imgproc.morphologyEx(auxiliar,auxiliar,Imgproc.MORPH_CLOSE,
							Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,new Size(11,11)));
			
			
			List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
			MatOfPoint2f mMOP2f1 = new MatOfPoint2f();
			Mat hierarchy = new Mat();
			
			Imgproc.findContours(auxiliar, contours, hierarchy, Imgproc.RETR_TREE, 
					Imgproc.CHAIN_APPROX_SIMPLE);
			
			int biggest = -1;
			double max_area = 0;
			Mat contour;
			double contourarea;
			double peri;
			
			for(int i=0;i<contours.size();++i)
			{
				contour = contours.get(i);
				contourarea = Imgproc.contourArea(contour);
				contours.get(i).convertTo(mMOP2f1, CvType.CV_32FC2);
				peri = Imgproc.arcLength(mMOP2f1, true);
				Imgproc.approxPolyDP(mMOP2f1,mMOP2f1,0.02*peri,true);
				if(contourarea > max_area /*&&  mMOP2f1.cols() == 4*/) 
				{
					max_area = contourarea;
					biggest = i;
				}
			}
			
			
			auxiliar.convertTo(auxiliar, CvType.CV_8UC4);
			Scalar s = new Scalar(255,0,0);
			if(biggest != -1)
				Imgproc.drawContours(auxiliar, contours, biggest, s, 4);
            
            
					
			//Imgproc.Sobel(auxiliar, auxiliar, CvType.CV_8U, 1, 0,3,1,0);
			//Imgproc.threshold(auxiliar, auxiliar, 0, 255, Imgproc.THRESH_OTSU|Imgproc.THRESH_BINARY_INV);
					
			//Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(64, 35));
			//Imgproc.morphologyEx(auxiliar, auxiliar, Imgproc.MORPH_CLOSE, element);
			
			Utils.matToBitmap(auxiliar, mBitmapPreview[0]); //Imagen original
			//Imgproc.medianBlur(auxiliar,auxiliar, 5);
			//Utils.matToBitmap(auxiliar, mBitmapPreview[1]); //Imagen blurred
			//Imgproc.Laplacian(auxiliar, auxiliar, CvType.CV_8U);
			//Utils.matToBitmap(auxiliar, mBitmapPreview[2]); //Imagen Laplacian
			//Mat otra = new Mat();
			//Imgproc.threshold(auxiliar, otra, -1, 255, 
			//	    Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);
			//Utils.matToBitmap(otra, mBitmapPreview[3]); //Imagen Threshold
			
			//Actualizar los Bitmaps
			runOnUiThread(new Runnable() {
			    public void run() {
			    	mimageView.setImageBitmap(mBitmapPreview[indice]);
			    	mimageView.setBackgroundColor(Color.BLACK);
			    }
			});
			
			return auxiliar;
	   // }
	   // else
	   // {
	   // 	return inputFrame;
	   // }
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

}
