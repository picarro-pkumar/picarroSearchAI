// Test localStorage functionality
console.log('Testing localStorage...');

// Test if localStorage is available
try {
  const test = '__localStorage_test__';
  localStorage.setItem(test, test);
  localStorage.removeItem(test);
  console.log('✅ localStorage is available');
} catch (error) {
  console.error('❌ localStorage is not available:', error);
}

// Test saving and loading data
try {
  const testData = {
    chats: [
      {
        id: 'test-1',
        title: 'Test Chat',
        messages: [
          {
            id: 'msg-1',
            content: 'Hello',
            role: 'user',
            timestamp: new Date().toISOString()
          }
        ],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
    ]
  };
  
  localStorage.setItem('picarro-chats', JSON.stringify(testData));
  console.log('✅ Successfully saved test data to localStorage');
  
  const loadedData = localStorage.getItem('picarro-chats');
  if (loadedData) {
    const parsed = JSON.parse(loadedData);
    console.log('✅ Successfully loaded data from localStorage:', parsed);
  } else {
    console.log('❌ No data found in localStorage');
  }
} catch (error) {
  console.error('❌ Error testing localStorage:', error);
} 