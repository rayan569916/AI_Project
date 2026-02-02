import { Component, ElementRef, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Attachment } from './chat.section.component.interface';
import { ChatSectionService } from './chat.section.service';

@Component({
  selector: 'app-chat.section.component',
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.section.component.html',
  styleUrl: './chat.section.component.css',
})
export class ChatSectionComponent {
  chatValue: string = ""
  selectedFiles: Attachment[] = []
  showCard: boolean = false;
  showModelDetail: boolean = false;
  FileSelected: boolean = false;
  borderRadius = "rounded-full items-center";
  isLoading = false;
  selectedModel:string="fast";
  chat_history_array:[]=[]
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>
  @ViewChild('chatTextarea') chatTextarea!: ElementRef<HTMLTextAreaElement>

  constructor(private service : ChatSectionService){}

  showCardMethod() {
    this.showCard = !this.showCard;
  }
  showModelDetailMethod(){
    this.showModelDetail = !this.showModelDetail;   
  }

  selectCloseMethod(){
    if (this.showCard) this.showCard=false;
    if (this.showModelDetail) this.showModelDetail=true;
  }

  modelSelecter(type:string){
    this.selectedModel=type;
    this.showModelDetailMethod()
  }

  fileSelecter(type: string) {
    const input = this.fileInput.nativeElement;
    if (type == "document") {
      input.accept = '.pdf,.doc,.docx,.xls,.xlsx,.ppt,.pptx,.txt'
    }
    else if (type == "image") {
      input.accept = 'image/*';
    }
    input.click()
  }

  onFileSelected(event: Event) {
    const input = this.fileInput.nativeElement;
    if (!input.files || input.files.length == 0) return;

    Array.from(input.files).forEach(element => {
      if (element.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = () => {
          this.selectedFiles.push({
            file: element,
            preview: reader.result as string
          })
        }
        reader.readAsDataURL(element);
      }
      else {
        this.selectedFiles.push({ file: element })
      }
      this.FileSelected = true;
      this.showCard = false;
      this.autoGrowForFile()
    });
  }

  autoGrow(textarea: HTMLTextAreaElement) {
    textarea.style.height = 'auto';
    const maxHeight = 300;

    if (textarea.scrollHeight <= maxHeight) {
      textarea.style.overflowY = 'hidden';
      textarea.style.height = textarea.scrollHeight + 'px';
      this.autoGrowForFile();
    }
    else if (textarea.scrollHeight >= maxHeight) {
      textarea.style.overflowY = 'auto';
      textarea.style.height = textarea.scrollHeight + 'px';
      this.autoGrowForFile();
    }
    else {
      textarea.style.overflowY = 'auto';
    }
  }

  autoGrowForFile() {
    this.borderRadius = "rounded-lg items-end";
  }
  removeItem(file: Attachment) {
    this.selectedFiles = this.selectedFiles.filter(f => f != file);
  }


  sendChatMethod() {
    const formData = new FormData();
    formData.append('user_chat', this.chatValue);
    formData.append('selected_mode',this.selectedModel);
    this.selectedFiles.forEach((item,index)=>{
      formData.append('attachments',item.file)
    })
    formData.append('chat_history',JSON.stringify(this.chat_history_array))
    this.isLoading = true;
    this.chatValue = "";
    this.selectedFiles = [];
    this.FileSelected = false;
    this.borderRadius = "rounded-full items-center"; 

    this.service.sendChatMethod(formData).subscribe(res=>{
      console.log(res)
    })
  }
}
