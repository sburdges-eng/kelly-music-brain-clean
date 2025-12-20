import React, { useState } from 'react';
import './EmotionWheel.css';

interface EmotionData {
  emotions: {
    [key: string]: {
      intensities: {
        [key: string]: string[];
      };
    };
  };
}

export interface SelectedEmotion {
  base: string;
  intensity: string;
  sub: string;
}

interface EmotionWheelProps {
  emotions: EmotionData | null;
  onEmotionSelected: (emotion: SelectedEmotion | null) => void;
}

const getEmotionColor = (base: string): string => {
  const colorMap: { [key: string]: string } = {
    angry: 'emotion-angry',
    happy: 'emotion-happy',
    sad: 'emotion-sad',
    fear: 'emotion-fear',
    disgust: 'emotion-disgust',
    surprise: 'emotion-surprise',
    neutral: 'emotion-neutral'
  };
  return colorMap[base.toLowerCase()] || 'emotion-neutral';
};

export const EmotionWheel: React.FC<EmotionWheelProps> = ({ emotions, onEmotionSelected }) => {
  const [selectedBase, setSelectedBase] = useState<string | null>(null);
  const [selectedIntensity, setSelectedIntensity] = useState<string | null>(null);
  const [selectedSub, setSelectedSub] = useState<string | null>(null);

  if (!emotions) {
    return (
      <div className="emotion-wheel-empty">
        Load emotions to begin selection
      </div>
    );
  }

  const baseEmotions = Object.keys(emotions.emotions);

  const handleBaseClick = (base: string) => {
    setSelectedBase(base);
    setSelectedIntensity(null);
    setSelectedSub(null);
    onEmotionSelected(null);
  };

  const handleIntensityClick = (intensity: string) => {
    setSelectedIntensity(intensity);
    setSelectedSub(null);
    onEmotionSelected(null);
  };

  const handleSubClick = (sub: string) => {
    setSelectedSub(sub);
    if (selectedBase && selectedIntensity) {
      onEmotionSelected({
        base: selectedBase,
        intensity: selectedIntensity,
        sub: sub
      });
    }
  };

  const resetSelection = () => {
    setSelectedBase(null);
    setSelectedIntensity(null);
    setSelectedSub(null);
    onEmotionSelected(null);
  };

  const getIntensities = () => {
    if (!selectedBase) return [];
    return Object.keys(emotions.emotions[selectedBase].intensities);
  };

  const getSubEmotions = () => {
    if (!selectedBase || !selectedIntensity) return [];
    return emotions.emotions[selectedBase].intensities[selectedIntensity] || [];
  };

  return (
    <div className="emotion-wheel-container">
      {selectedBase && selectedIntensity && selectedSub && (
        <div className="emotion-wheel-selected animate-fadeIn">
          <div className="emotion-wheel-selected-content">
            <div className="emotion-wheel-selected-label">Selected Emotion:</div>
            <div className="emotion-wheel-selected-value">
              {selectedBase} → {selectedIntensity} → {selectedSub}
            </div>
          </div>
          <button
            onClick={resetSelection}
            className="emotion-wheel-clear-btn"
          >
            Clear
          </button>
        </div>
      )}

      <div className="emotion-wheel-step">
        <h4 className="emotion-wheel-step-title">
          {selectedBase ? '1. Base Emotion (selected)' : '1. Select Base Emotion'}
        </h4>
        <div className="emotion-wheel-grid emotion-wheel-grid-base">
          {baseEmotions.map(base => (
            <button
              key={base}
              onClick={() => handleBaseClick(base)}
              className={`emotion-wheel-btn ${getEmotionColor(base)} ${selectedBase === base ? 'emotion-wheel-btn-selected' : ''}`}
            >
              {base}
            </button>
          ))}
        </div>
      </div>

      {selectedBase && (
        <div className="emotion-wheel-step animate-fadeIn">
          <h4 className="emotion-wheel-step-title">
            {selectedIntensity ? '2. Intensity Level (selected)' : '2. Select Intensity Level'}
          </h4>
          <div className="emotion-wheel-grid emotion-wheel-grid-intensity">
            {getIntensities().map(intensity => (
              <button
                key={intensity}
                onClick={() => handleIntensityClick(intensity)}
                className={`emotion-wheel-btn ${getEmotionColor(selectedBase)} ${selectedIntensity === intensity ? 'emotion-wheel-btn-selected' : ''}`}
              >
                {intensity}
              </button>
            ))}
          </div>
        </div>
      )}

      {selectedBase && selectedIntensity && (
        <div className="emotion-wheel-step animate-fadeIn">
          <h4 className="emotion-wheel-step-title">
            {selectedSub ? '3. Specific Emotion (selected)' : '3. Select Specific Emotion'}
          </h4>
          <div className="emotion-wheel-grid emotion-wheel-grid-sub">
            {getSubEmotions().map(sub => (
              <button
                key={sub}
                onClick={() => handleSubClick(sub)}
                className={`emotion-wheel-btn ${getEmotionColor(selectedBase)} ${selectedSub === sub ? 'emotion-wheel-btn-selected' : ''}`}
              >
                {sub}
              </button>
            ))}
          </div>
        </div>
      )}

      {!selectedBase && (
        <div className="emotion-wheel-empty">
          Choose your emotional starting point above
        </div>
      )}
    </div>
  );
};
