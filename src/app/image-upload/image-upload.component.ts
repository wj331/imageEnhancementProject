import { Component, ChangeDetectorRef, OnInit} from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Observable, EMPTY, throwError , from, forkJoin, of, isObservable, combineLatest } from 'rxjs';
import { NzCardModule } from 'ng-zorro-antd/card';
import { NzTagModule } from 'ng-zorro-antd/tag';
import { NzFormModule } from 'ng-zorro-antd/form';
import { NzInputModule } from 'ng-zorro-antd/input';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzDatePickerModule } from 'ng-zorro-antd/date-picker';
import { concatMap, tap, catchError, switchMap, map, shareReplay } from 'rxjs/operators';
interface ImprovementResult {
  new_detections: { label: string; confidence: number; position: [number, number, number, number] }[];
  lost_detections: { label: string; confidence: number; position: [number, number, number, number] }[];
  label_changes: {
    original: { label: string; confidence: number; position: [number, number, number, number] };
    brightened: { label: string; confidence: number; position: [number, number, number, number] };
    spatial_similarity: number;
  }[];
  confidence_changes: {
    label: string;
    original_confidence: number;
    new_confidence: number;
    change: number;
    position: [number, number, number, number];
    spatial_similarity: number;
  }[];
  summary: {
    total_improvements: number;
    total_degradations: number;
    total_confidence_changes: number;
    average_confidence_changes: number;
    new_detections: number;
    lost_detections: number;
    label_changes: number;
  };
}
interface AggregatedImprovement {
  averageConfidenceChange: number;
  totalConfidenceChange: number;
  totalImprovement: number;
  totalDegradation: number;
  newDetections: number;
  lostDetections: number;
  labelChanges: number;
}


@Component({
  selector: 'app-image-upload',
  standalone: true,
  imports: [CommonModule, NzFormModule, NzInputModule, NzButtonModule, NzDatePickerModule, NzCardModule, NzTagModule],
  templateUrl: './image-upload.component.html',
  styleUrls: ['./image-upload.component.css']
})
export class ImageUploadComponent{

  uploadedImages: { path: string, detectionResults: any[] }[] = [];
  brightenedImages: { path: string, detectionResults: any[] }[] = [];
  imageError: string | null = null;
  detectionResultsOriginal: { x: number; y: number; width: number; height: number; label: string; confidence: number; }[] | null = null;
  detectionResultsBrightened: { x: number; y: number; width: number; height: number; label: string; confidence: number; }[] | null = null;
  brightenedImagePath: string | null = null;
  uploadedImagePath: string | null = null;
  detectedImageUrl: string | null = null;
  backendUrl = 'http://localhost:5000/';
  scaleX: number = 1;
  scaleY: number = 1;
  aggregatedImprovement: any = {
    averageConfidenceChanges: 0,
    totalConfidenceChanges: 0,
    totalImprovement: 0,
    totalDegradation: 0,
    newDetections: 0,
    lostDetections: 0,
    labelChanges: 0,
  };
  

  constructor(private http: HttpClient, private cdRef: ChangeDetectorRef) {}
    onFileSelected(event: Event) {
      const files = (event.target as HTMLInputElement).files;
    
      if (files && files.length > 0) {
        const imageFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
    
        if (imageFiles.length === 0) {
          this.imageError = 'No valid image files found in the selected folder.';
          console.error(this.imageError);
          return;
        }
    
        from(imageFiles).pipe(
          concatMap((file) => this.processImage(file).pipe(
            catchError((error) => {
              console.error('Error during file processing:', error);
              return EMPTY;  // Continue processing other files even if one fails
            })
          ))
        ).subscribe({
          complete: () =>  console.log('All files processed successfully.')
        });
      } else {
        this.imageError = 'No files selected.';
        console.error(this.imageError);
      }
    }
    
    processImage(file: File) {
      const formData = new FormData();
      formData.append('image', file);
        
      // Step-by-step processing pipeline
      return this.http.post<{ filePath: string }>(`${this.backendUrl}upload`, formData).pipe(
        switchMap(response => {
          console.log('File uploaded', response.filePath);
          this.uploadedImages.push({ path: response.filePath, detectionResults: [] });
        return forkJoin({
          originalDetection: this.performObjectDetection(response.filePath, true),
          brightenedPath: this.brighten(response.filePath)
        }).pipe(
          switchMap(({ originalDetection, brightenedPath }) => {
            console.log('Brightened image path:', brightenedPath);
            
            this.brightenedImages.push({ path: brightenedPath, detectionResults: []});

            return this.performObjectDetection(brightenedPath, false).pipe(
              tap(brightenedDetection => {
                console.log('Detection results for brightened image:', brightenedDetection);
              })
            );
          })
        );
      }),
      catchError(error => {
        console.error('Error processing image:', file.name, error);
        return EMPTY;
      })
    )
  }
    
    performObjectDetection(imagePath: string, isOriginal: boolean) {
      const requestBody = { filePath: imagePath };
    
      return this.http.post<{ detections: { x: number, y: number, width: number, height: number, label: string, confidence: number }[] }>(
        `${this.backendUrl}detect`,
        requestBody
      ).pipe(
        tap((response) => {
          if (isOriginal) {
            const image = this.uploadedImages.find(img => img.path === imagePath);
            if (image) {
              image.detectionResults = response.detections;
              this.cdRef.detectChanges();
              console.log('Original detection results:', this.detectionResultsOriginal);
            }
          } else {
            const image = this.brightenedImages.find(img => img.path === imagePath);
            if (image) {
              image.detectionResults = response.detections;
              this.cdRef.detectChanges();
              console.log('Brightened detection results:', this.detectionResultsBrightened);
            }
          }
        })
      );
    }
    
    brighten(imagePath: string) {
      const requestBody = { filePath: imagePath };
    
      return this.http.post<{ enhanced_image_path: string }>(
        `${this.backendUrl}brighten`,
        requestBody
      ).pipe(
        map((response : {enhanced_image_path: string}) => response.enhanced_image_path),  // Return the path of the brightened image
        tap((brightenedPath) => console.log('Brightened image url:', brightenedPath)),
        catchError((error) => {
          console.error('Error during image brightening:', error);
          return EMPTY;
        })
      );
    }

    getBrightenedDetections(originalImagePath: string) {
      const imageFileName = originalImagePath.split('/').pop() || '';

      const brightenedImage = this.brightenedImages.find(img => img.path.includes(imageFileName));

      return brightenedImage?.detectionResults || [];
    }

    getBrightenedImagePath(uploadedImagePath: string): string | undefined{
      const fileName = uploadedImagePath.split('/').pop() || '';
      return this.brightenedImages.find(img => img.path.includes(fileName))?.path;
    }


    calculateImprovement(uploadedImage: any): Observable<ImprovementResult> {
      const uploadedDetections = uploadedImage.detectionResults || [];
    
      // Wait for brightened detections
      const brightenedDetections = this.getBrightenedDetections(uploadedImage.path);
      const brightenedDetectionsList = Array.isArray(brightenedDetections) ? brightenedDetections : [];
    
      const requestBody = {
        uploadedDetections,
        brightenedDetections: brightenedDetectionsList
      };
    
      console.log("Sending request:", requestBody);
    
      let result: ImprovementResult | undefined;

      return this.http.post<ImprovementResult>(
        `${this.backendUrl}calculate-improvement`,
        requestBody
      ).pipe(
        catchError(error => {
          console.error("error during improvement calculation", error);
          return of(this.getDefaultImprovementResult());
        })
      )
    }
  
    getDefaultImprovementResult(): ImprovementResult {
      return {
        new_detections: [],
        lost_detections: [],
        label_changes: [],
        confidence_changes: [],
        summary: {
          total_improvements: 0,
          total_degradations: 0,
          total_confidence_changes: 0,
          average_confidence_changes: 0,
          new_detections: 0,
          lost_detections: 0,
          label_changes: 0
        }
      };
    }

    improvementResultMap: Map<string, ImprovementResult> = new Map<string, ImprovementResult>(); // Store direct results
    loadingStates: { [imagePath: string]: boolean } = {};

    getImprovementForImage(image: { path: string, detectionResults: any[] }) {
      if (this.improvementResultMap.has(image.path)) {
        return this.improvementResultMap.get(image.path);
      }

      if (this.loadingStates[image.path]) {
        return undefined;
      }
      if (!image.detectionResults || !this.getBrightenedDetections(image.path)?.length) {
        console.log("Detections not ready yet, skipping improvement calculation.");
        this.retryImprovementCheck(image)
        return undefined;
      }
      this.loadingStates[image.path] = true;
      this.calculateImprovement(image).subscribe((improvement) => {
        this.improvementResultMap.set(image.path, improvement);
        this.loadingStates[image.path] = false;
        console.log("improvement for this image is obtained", improvement);
        this.calculateAggregatedImprovement(improvement);
      });
      return undefined;
    }

    retryImprovementCheck(image: { path: string, detectionResults: any[] }) {
      if (this.loadingStates[image.path]) return;

      this.loadingStates[image.path] = true;
      const retryInterval = setInterval(() => {
        if (image.detectionResults && this.getBrightenedDetections(image.path)?.length) {
          clearInterval(retryInterval);

          console.log("detections are now ready, calculating improvement...");
          this.calculateImprovement(image).subscribe((improvement) => {
            this.improvementResultMap.set(image.path, improvement);
            this.loadingStates[image.path] = false;
            console.log("improvement for this image is obtained", improvement);
            this.calculateAggregatedImprovement(improvement);
          });
          this.getImprovementForImage(image);
        }
      }, 5000)
    }
    calculateAggregatedImprovement(improvement : ImprovementResult) {
      this.aggregatedImprovement.averageConfidenceChanges += (improvement.summary.average_confidence_changes * 100);
      this.aggregatedImprovement.totalConfidenceChanges += (improvement.summary.total_confidence_changes * 100);

      if (improvement.summary.total_confidence_changes > 0) {
        this.aggregatedImprovement.totalImprovement += 1;
      } else if (improvement.summary.total_confidence_changes < 0) {
        this.aggregatedImprovement.totalDegradation += 1;
      }
      if (improvement.summary.new_detections > 0) {
        this.aggregatedImprovement.newDetections += 1;
      }
      if (improvement.summary.lost_detections > 0) {
        this.aggregatedImprovement.lostDetections += 1;
      }
      if (improvement.summary.label_changes > 0) {
        this.aggregatedImprovement.labelChanges += 1;
      }
    }
}

