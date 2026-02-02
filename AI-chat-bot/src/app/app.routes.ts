import { Routes } from '@angular/router';

export const routes: Routes = [
    {path:'',redirectTo: 'chat', pathMatch: 'full'},
    {path:'chat',loadComponent:()=>import('./chat.section.component/chat.section.component').then(m=>m.ChatSectionComponent) }
];
