import { HttpClient } from "@angular/common/http";
import {baseURL} from './chat.section.config'
import { Observable } from "rxjs";
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})

export class ChatSectionService{
    constructor(private http:HttpClient) {}

    sendChatMethod(formData:FormData): Observable<any>{
        return this.http.post(`${baseURL}/send_chat`,formData);
    }
}