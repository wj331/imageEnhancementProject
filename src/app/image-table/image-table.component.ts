import { Component, Input} from '@angular/core';
import { Observable, of } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { HttpClient } from '@angular/common/http';
import { CommonModule} from '@angular/common';
import { NzTagModule } from 'ng-zorro-antd/tag'; // Import Ant Design Tag Module



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

@Component({
  selector: 'app-image-table',
  standalone: true,
  imports: [CommonModule, NzTagModule],
  templateUrl: './image-table.component.html',
  styleUrl: './image-table.component.css'
})


export class ImageTableComponent {
  @Input() uploadedImages: { path: string, detectionResults: any[] }[] = [];
  @Input() brightenedImages: { path: string, detectionResults: any[] }[] = [];
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
  constructor(private http: HttpClient) { }

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
    let attempts = 0;
    const maxAttempts = 4;

    const retryInterval = setInterval(() => {
      attempts ++;
      if ((image.detectionResults && this.getBrightenedDetections(image.path)?.length) || (attempts >= maxAttempts)) {
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

  getImageScale(imagePath: string, imgElement: HTMLImageElement) {
    const originalWidth = imgElement.naturalWidth; // Original image width
    const originalHeight = imgElement.naturalHeight; // Original image height
  
    const displayedWidth = imgElement.clientWidth; // Displayed image width
    const displayedHeight = imgElement.clientHeight; // Displayed image height
  
    const scaleX = displayedWidth / originalWidth;
    const scaleY = displayedHeight / originalHeight;
  
    return { scaleX, scaleY };
  }
  
  onImageLoad(image: any, imagePath: string) {
    const scales = this.getImageScale(imagePath, image.target);
    this.scaleX = scales.scaleX;
    this.scaleY = scales.scaleY;
  }  
}
