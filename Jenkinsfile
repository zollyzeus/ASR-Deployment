#!groovy
pipeline {
    agent any

    stages {
        stage('Init') {
            steps {
                    dir('D:\\CDAC\\AI_Trends\\Docker_Projects\\ASR-Deployment') {
                    // some block
                    //bat 'docker stop $(docker ps -q)'
                }
            }
        }
        stage('Clone Git Repository') {               
            steps {
                dir('D:\\CDAC\\AI_Trends\\Docker_Projects\\ASR-Deployment') {
                    // some block
                    bat 'git pull origin b1'    
                }
            }
        } 
        stage('Build Docker Image') {         
            steps {
                dir('D:\\CDAC\\AI_Trends\\Docker_Projects\\ASR-Deployment') {
                    // some block                    
                    bat 'docker build -t zollyzeus/asrproject:latest .'
                }
            }
        }   
        stage('Push Docker Image') {         
            steps {       
                dir('D:\\CDAC\\AI_Trends\\Docker_Projects\\ASR-Deployment') {
                    // some block
                    bat 'docker push zollyzeus/asrproject:latest'
                }
            }
        }        
    }
}