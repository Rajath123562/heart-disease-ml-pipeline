import React from 'react';
import {
  X,
  Heart,
  Award,
  BookOpen,
  Code,
  ShieldCheck,
  CheckCircle2
} from 'lucide-react';

interface AboutModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const AboutModal: React.FC<AboutModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/60 backdrop-blur-xs animate-fadeIn">
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl max-w-xl w-full p-6 sm:p-8 shadow-2xl space-y-6 relative max-h-[90vh] overflow-y-auto">
        <button
          onClick={onClose}
          className="absolute top-5 right-5 p-2 rounded-full text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition cursor-pointer"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-2xl bg-rose-500/10 text-rose-500">
            <Heart className="w-6 h-6 fill-rose-500" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-900 dark:text-white">
              Heart Disease ML Pipeline
            </h3>
            <p className="text-xs text-slate-500">University & Internship Machine Learning Capstone (Week 4)</p>
          </div>
        </div>

        <div className="space-y-4 text-xs sm:text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
          <p>
            This single-page application is a complete, production-quality implementation of a machine learning workflow for early cardiovascular risk prediction.
          </p>

          <div className="space-y-2">
            <h4 className="font-bold text-slate-900 dark:text-white flex items-center gap-1.5 text-xs uppercase tracking-wider">
              <Code className="w-4 h-4 text-rose-500" />
              <span>Zero External ML Libraries</span>
            </h4>
            <p className="text-xs text-slate-500">
              All machine learning algorithms (Batch Gradient Descent Logistic Regression, CART Decision Tree with Gini/Entropy criteria, ROC-AUC trapezoidal integration, Stratified K-Fold CV, and scalers) were implemented <strong>from scratch in native TypeScript</strong>. The entire application runs 100% offline in the browser without backend servers or API keys.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="font-bold text-slate-900 dark:text-white flex items-center gap-1.5 text-xs uppercase tracking-wider">
              <Award className="w-4 h-4 text-indigo-500" />
              <span>Dataset Provenance</span>
            </h4>
            <p className="text-xs text-slate-500">
              Benchmark Cleveland Heart Disease Database collected by Dr. Robert Detrano, V.A. Medical Center, Long Beach and Cleveland Clinic Foundation. Distributed via Kaggle and the UCI Machine Learning Repository.
            </p>
          </div>

          <div className="p-3 bg-amber-50 dark:bg-amber-950/40 rounded-xl border border-amber-500/30 text-amber-800 dark:text-amber-300 text-xs">
            <strong>Clinical Safety Disclaimer:</strong> This application is intended solely for educational, academic, and demonstration purposes. It does not provide medical diagnoses or replace physician care.
          </div>
        </div>

        <div className="pt-2 flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2.5 bg-slate-900 hover:bg-slate-800 dark:bg-white dark:hover:bg-slate-100 text-white dark:text-slate-900 font-semibold text-xs rounded-xl transition cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
