pipeline {
    agent any

    environment {
       CXX = "nvcc"
       LD = "nvcc"
    }

    stages {
        stage ('git'){
            steps {
                checkout([
                    $class: 'GitSCM',
                    branches: scm.branches,
                    doGenerateSubmoduleConfigurations: false,
                    extensions: scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: false, recursiveSubmodules: true, reference: '', trackingSubmodules: false]],
                    submoduleCfg: [],
                    userRemoteConfigs: scm.userRemoteConfigs])
            }
        }

        stage ('pre-analysis') {
            steps {
                sh 'cppcheck --xml-version=2 --enable=all --std=c++11 include/cpp_utils/*.hpp test/src/*.cpp 2> cppcheck_report.xml'
                sh 'sloccount --duplicates --wide --details include/cpp_utils/*.hpp test/src/*.cpp > sloccount.sc'
                sh 'cccc include/cpp_utils/*.hpp test/*.cpp || true'
            }
        }

        stage ('build'){
            steps {
                sh 'make clean'
                sh 'make -j6 release'
            }
        }

        stage ('test'){
            steps {
                sh 'PATH="$PATH:/opt/cuda/bin/" ./scripts/test_runner.sh'
                archive 'catch_report.xml'
                junit 'catch_report.xml'
            }
        }

        stage ('sonar-master'){
            when {
                branch 'master'
            }
            steps {
                sh "/opt/sonar-runner/bin/sonar-runner"
            }
        }

        stage ('sonar-branch'){
            when {
                not {
                    branch 'master'
                }
            }
            steps {
                sh "/opt/sonar-runner/bin/sonar-runner -Dsonar.branch=${env.BRANCH_NAME}"
            }
        }
    }

    post {
        always {
            script {
                if (currentBuild.result == null) {
                    currentBuild.result = 'SUCCESS'
                }
            }

            step([$class: 'Mailer',
                notifyEveryUnstableBuild: true,
                recipients: "baptiste.wicht@gmail.com",
                sendToIndividuals: true])
        }
    }
}
