// 'use client';

// import { useAuth0 } from '@auth0/auth0-react';
// import ProtectedRoute from '@/components/auth/ProtectedRoute';
// import ReactorCoreCard from '@/components/systems/ReactorCoreCard';
// import PumpSystemCard from '@/components/systems/PumpSystemCard';
// import TurbineSystemCard from '@/components/systems/TurbineSystemCard';
// import CoolantSystemCard from '@/components/systems/CoolantSystemCard';

// export default function Home() {
//   const { isAuthenticated } = useAuth0();

//   return (
//     <div className="space-y-6">
//       {!isAuthenticated ? (
//         <div className="card">
//           <h2 className="text-xl font-semibold mb-4">Welcome to Nuclear Safety System</h2>
//           <p className="text-[var(--text-secondary)] mb-4">
//             Please log in to access the monitoring dashboard and sensor data.
//           </p>
//         </div>
//       ) : (
//         <ProtectedRoute>
//           <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
//             <div className="lg:col-span-2">
//               <ReactorCoreCard />
//             </div>
//             <div className="lg:col-span-2">
//               <PumpSystemCard />
//             </div>
//             <div className="lg:col-span-2">
//               <TurbineSystemCard />
//             </div>
//             <div className="lg:col-span-2">
//               <CoolantSystemCard />
//             </div>
//           </div>
//         </ProtectedRoute>
//       )}
//     </div>
//   );
// }

'use client';

import { useAuth } from '@/hooks/useAuth';
import OperatorDashboard from '@/components/operator/OperatorDashboard';
import ComplianceHome from '@/components/compliance/ComplianceHome';
import ProtectedRoute from '@/components/auth/ProtectedRoute';

export default function Home() {
  const { isAuthenticated, isOperator, isCompliance } = useAuth();

  return (
    <ProtectedRoute>
      <div className="container mx-auto px-4 py-6">
        {!isAuthenticated && (
          <div className="card max-w-2xl mx-auto mt-8 p-6">
            <h2 className="text-2xl font-bold mb-4">
              Welcome to Nuclear Safety System
            </h2>
            <p>Please log in to access the system.</p>
          </div>
        )}

        {isAuthenticated && (
          <>
            {isOperator && <OperatorDashboard />}
            {isCompliance && <ComplianceHome />}
          </>
        )}
      </div>
    </ProtectedRoute>
  );
}