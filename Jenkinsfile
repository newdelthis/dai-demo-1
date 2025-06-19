pipeline {
    agent any
    stages {
        stage('Clone') {
            steps {
                git 'https://github.com/newdelthis/dai-demo-1.git'
            }
        }
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t salary-api .'
            }
        }
        stage('Run Container') {
            steps {
                sh 'docker run -d -p 5000:5000 salary-api'
            }
        }
    }
}
